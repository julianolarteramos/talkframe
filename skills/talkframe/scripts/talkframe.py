#!/usr/bin/env python3
"""talkframe: turn a portrait image + a voice track into a talking video (InfiniteTalk on a rented GPU).

Subcommands
  check     verify local tools and the chosen backend (no GPU time is spent)
  generate  plan, render on the backend, stitch, and write <out>.mp4 + <out>.report.json

Backends
  colab   Google Colab via the official `colab` CLI (google-colab-cli). Needs Colab Pro/compute units.
  ssh     any Linux host with an NVIDIA GPU and CUDA PyTorch, reached with `ssh user@host`.
  local   this machine, if it has an NVIDIA GPU and CUDA PyTorch.

Only the Python standard library plus ffmpeg/ffprobe are needed locally.
"""
import argparse
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REMOTE_JOB = os.path.join(HERE, "remote", "job.py")
REMOTE_FILES = [REMOTE_JOB, os.path.join(HERE, "remote", "longcat.py")]  # everything the GPU machine needs
PIXELS = {480: 832 * 480, 720: 1280 * 720}
DEFAULT_PROMPT = (
    "A person speaks directly to the camera with natural, expressive facial expressions, subtle head movements and blinking, "
    "gesturing naturally with their hands while explaining. Their hair moves slightly. Realistic, steady medium shot."
)
COLAB_REMOTE = "/content/talkframe"
FPS = 25
EXIT_NO_GPU = 75  # EX_TEMPFAIL: no GPU was free; safe to retry later (queues can treat it as "busy", not "broken")


# ---------------------------------------------------------------- helpers
def log(msg):
    print(f"[talkframe] {msg}", flush=True)


def run(cmd, check=True, capture=True, timeout=None):
    r = subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout)
    if check and r.returncode:
        raise SystemExit(f"command failed ({r.returncode}): {' '.join(map(str, cmd))[:200]}\n{(r.stderr or '')[-2000:]}")
    return r


def probe(path):
    r = run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height", "-show_entries", "format=duration",
             "-of", "json", path])
    return json.loads(r.stdout)


def image_size(path):
    for s in probe(path).get("streams", []):
        if s.get("codec_type") == "video" and s.get("width"):
            return s["width"], s["height"]
    raise SystemExit(f"not an image: {path}")


def audio_duration(path):
    d = float(probe(path).get("format", {}).get("duration") or 0)
    if d <= 0:
        raise SystemExit(f"no audio duration: {path}")
    return d


def target_size(w, h, resolution):
    """Keep the image's aspect ratio, ~same pixel count as the model's training bucket, multiples of 16."""
    aspect = w / h
    tw = math.sqrt(PIXELS[resolution] * aspect)
    th = tw / aspect
    return int(round(tw / 16) * 16), int(round(th / 16) * 16)


def silences(path):
    r = run(["ffmpeg", "-hide_banner", "-i", path, "-af", "silencedetect=noise=-35dB:d=0.2", "-f", "null", "-"], check=False)
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", r.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r.stderr)]
    return [(s + e) / 2 for s, e in zip(starts, ends)]


def cut_points(path, duration, max_seg):
    """Split long audio at pauses so each piece is <= max_seg seconds (InfiniteTalk drifts in color past ~1 min)."""
    if duration <= max_seg:
        return []
    mids, cuts, start = silences(path), [], 0.0
    while duration - start > max_seg:
        limit = start + max_seg
        options = [m for m in mids if start + max_seg * 0.5 < m <= limit]
        cut = max(options) if options else limit
        cuts.append(round(cut, 3))
        start = cut
    return cuts


# ---------------------------------------------------------------- planning
def plan(args):
    for p in (args.image, args.audio):
        if not os.path.isfile(p):
            raise SystemExit(f"file not found: {p}")
    iw, ih = image_size(args.image)
    if args.size:
        w, h = (int(x) for x in args.size.lower().split("x"))
        w, h = w // 16 * 16, h // 16 * 16
    else:
        w, h = target_size(iw, ih, args.resolution)
    if args.max_segment < 5:
        raise SystemExit("--max-segment must be at least 5 seconds")
    dur = audio_duration(args.audio)
    cuts = cut_points(args.audio, dur, args.max_segment)
    out = os.path.abspath(args.out)
    work = os.path.abspath(args.workdir or (os.path.splitext(out)[0] + ".talkframe"))
    try:
        os.makedirs(os.path.join(work, "in"), exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=os.path.join(work, "in")):
            pass  # an existing dir can still be read-only
    except OSError as e:
        if not args.dry_run:
            raise SystemExit(f"cannot write the work dir {work}: {e}")
        # a dry run only needs scratch space; read-only sandboxes still get a plan
        bad, work = work, tempfile.mkdtemp(prefix="talkframe-plan-")
        os.makedirs(os.path.join(work, "in"))
        log(f"cannot write {bad} ({e}); planning in {work} instead")

    # Framing. If the photo's shape differs from the video's, crop the photo here, before the model sees it. When the
    # video is wider, keep a horizontal band centred at --crop-y of the photo's height (0.30 suits most portraits:
    # a plain centre crop of a vertical photo into a landscape strip lands on the chin or neck). When it is taller,
    # keep the horizontal centre. The model then gets an image with the right shape and only scales it.
    ext = os.path.splitext(args.image)[1].lower() or ".png"
    src_ar, dst_ar = iw / ih, w / h
    if abs(src_ar / dst_ar - 1) > 0.02:
        if dst_ar > src_ar:
            cw, ch = iw, int(round(iw / dst_ar))
            cx, cy = 0, int(min(max(args.crop_y * ih - ch / 2, 0), ih - ch))
        else:
            cw, ch = int(round(ih * dst_ar)), ih
            cx, cy = (iw - cw) // 2, 0
        crop = [cx, cy, cw, ch]
        ext = ".png"
        run(["ffmpeg", "-v", "error", "-y", "-i", args.image, "-vf", f"crop={cw}:{ch}:{cx}:{cy}", "-frames:v", "1",
             os.path.join(work, "in", "image.png")])
    else:
        crop = [0, 0, iw, ih]
        shutil.copy(args.image, os.path.join(work, "in", "image" + ext))
    bounds = [0.0, *cuts, dur]
    jobs = []
    for i in range(len(bounds) - 1):
        name = f"seg{i:03d}"
        wav = os.path.join(work, "in", f"{name}.wav")
        # The model only hears this copy; the final video keeps the original audio. Pure digital silence (exact zeros,
        # typical between edited clips) makes InfiniteTalk open the mouth wide during pauses, so a pink-noise floor (~-57 dBFS RMS)
        # is mixed in (measured: ~20-30% less mouth opening in pauses). That is about the level of quiet room tone.
        noise = "" if args.no_noise_floor else \
            ";anoisesrc=color=pink:amplitude=0.008:sample_rate=24000[n];[v][n]amix=inputs=2:duration=first:normalize=0"
        run(["ffmpeg", "-v", "error", "-y", "-i", args.audio, "-filter_complex",
             f"[0:a]atrim={bounds[i]}:{bounds[i + 1]},asetpts=N/SR/TB,aformat=channel_layouts=mono,aresample=24000[v]{noise}",
             *([] if noise else ["-map", "[v]"]), "-ar", "24000", wav])
        jobs.append({"name": name, "image": "image" + ext, "audio": f"{name}.wav", "prompt": args.prompt or DEFAULT_PROMPT,
                     "width": w, "height": h, "steps": args.steps, "seed": args.seed, "audio_scale": args.audio_scale,
                     "seconds": round(bounds[i + 1] - bounds[i], 3),
                     "attention": args.attention,
                     "fp8_fast": args.fp8_fast, "model": args.model, "resolution": args.resolution,
                     "lc_mix_alpha": args.lc_mix_alpha, "lc_mix_steps": args.lc_mix_steps,
                     # whole frames this part must fill, from absolute frame positions so rounding never accumulates
                     "frames": round(bounds[i + 1] * FPS) - round(bounds[i] * FPS)})
    json.dump(jobs, open(os.path.join(work, "in", "jobs.json"), "w"), indent=1)
    p = {"image": os.path.abspath(args.image), "image_size": [iw, ih], "audio": os.path.abspath(args.audio),
         "audio_seconds": round(dur, 2), "audio_seconds_exact": dur, "output": out, "output_size": [w, h], "segments": len(jobs),
         "video_delay_frames": args.video_delay, "photo_crop": crop, "cut_points": cuts, "dry_run": bool(args.dry_run), "backend": args.backend, "workdir": work, "prompt": args.prompt or DEFAULT_PROMPT}
    json.dump(p, open(os.path.join(work, "plan.json"), "w"), indent=1)
    return p, jobs, work


# ---------------------------------------------------------------- backends
class Colab:
    def __init__(self, args):
        self.gpu, self.keep, self.attempts = args.gpu, getattr(args, "keep", False), args.gpu_attempts
        self.reuse = getattr(args, "session", None)
        self.s = self.reuse or f"talkframe-{time.strftime('%H%M%S')}"

    def c(self, *a, **kw):
        return run(["colab", "--auth", "adc", *a], **kw)

    def balance(self):
        r = self.c("usage", check=False)
        m = re.search(r"balance:\s*([\d.]+)", r.stdout + r.stderr)
        return float(m.group(1)) if m else None

    def check(self):
        if not shutil.which("colab"):
            return ["`colab` CLI not found: uv tool install google-colab-cli"]
        r = self.c("whoami", check=False)
        if r.returncode or "Email" not in r.stdout:
            return ["colab auth failed; run: gcloud auth application-default login --scopes=openid,"
                    "https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,"
                    "https://www.googleapis.com/auth/colaboratory"]
        b = self.balance()
        log(f"colab account ok, balance {b} compute units")
        return [] if (b or 0) >= 2 else [f"low Colab balance ({b} units); an A100 burns ~8/h"]

    def alive(self):
        return f"[{self.s}]" in self.c("sessions", check=False).stdout

    def start(self):
        # auth, CLI or balance problems are permanent: fail now instead of reporting them as "no GPU" (exit 75)
        problems = self.check()
        if problems:
            raise SystemExit("colab not usable: " + "; ".join(problems))
        if self.reuse:
            if not self.alive():
                raise SystemExit(f"--session {self.reuse}: no such running Colab session")
            log(f"reusing colab session {self.s}")
            return
        # --gpu may list fallbacks ("A100,H100,L4"): each attempt tries the next one, so a busy pool doesn't block the run
        gpus = [g.strip().upper() for g in self.gpu.split(",") if g.strip()]
        i = 0
        while i < self.attempts:
            gpu = gpus[i % len(gpus)]
            try:
                r = self.c("new", "-s", self.s, "--gpu", gpu, check=False, timeout=300)
                err = (r.stdout + r.stderr).lower()
            except subprocess.TimeoutExpired:
                err = "timeout"
            if self.alive():
                log(f"colab session {self.s} ({gpu})")
                return
            if "rejected accelerator" in err or "entitlement" in err:
                # this plan can't use that GPU at all (e.g. H100 without Pro+): drop it and try the others right away
                gpus.remove(gpu)
                log(f"this Colab plan can't use {gpu}; continuing with {','.join(gpus) or 'nothing'}")
                if not gpus:
                    raise SystemExit(f"none of the GPUs in --gpu {self.gpu} are allowed on this Colab plan")
                continue
            if not any(k in err for k in ("unavailable", "503", "timeout", "capacity", "try again")):
                raise SystemExit(f"colab new failed (not a busy pool): {err.strip()[-800:]}")
            i += 1
            if i < self.attempts:
                log(f"no {gpu} available, attempt {i}/{self.attempts}; retrying in 60s")
                time.sleep(60)
        self.c("stop", "-s", self.s, check=False)  # a timed-out `new` may still come up later; don't leave it billing
        log(f"could not get a Colab GPU ({self.gpu}) after {self.attempts} attempts; try later")
        sys.exit(EXIT_NO_GPU)

    def run_job(self, work, jobs):
        # one upload (inputs + job.py in a tarball) instead of one `colab upload` round trip per file
        import tarfile
        bundle = os.path.join(work, "bundle.tgz")
        with tarfile.open(bundle, "w:gz") as tf:
            tf.add(os.path.join(work, "in"), arcname="in")
            for f in REMOTE_FILES:
                tf.add(f, arcname=os.path.basename(f))
        p = subprocess.run(["colab", "--auth", "adc", "exec", "-s", self.s],
                           input=(f"import os, glob\nfor d in ('in', 'out'):\n    os.makedirs('{COLAB_REMOTE}/' + d, exist_ok=True)\n"
                                  f"    [os.remove(f) for f in glob.glob('{COLAB_REMOTE}/' + d + '/*') if os.path.isfile(f)]\n"),
                           text=True, capture_output=True)
        if p.returncode:
            raise SystemExit(f"colab exec failed: {p.stderr[-1500:]}")
        self.c("upload", "-s", self.s, bundle, f"{COLAB_REMOTE}/bundle.tgz", timeout=900)
        p = subprocess.run(["colab", "--auth", "adc", "exec", "-s", self.s], text=True, capture_output=True,
                           input=f"import tarfile\ntarfile.open('{COLAB_REMOTE}/bundle.tgz').extractall('{COLAB_REMOTE}')\n")
        if p.returncode:
            raise SystemExit(f"colab exec (unpack) failed: {p.stderr[-1500:]}")
        # exec keeps the kernel busy, so Colab does not reclaim the VM as idle; --timeout 0 allows long silent phases
        rc = subprocess.run(["colab", "--auth", "adc", "exec", "-s", self.s, "--timeout", "0", "-f", REMOTE_JOB]).returncode
        if rc:
            raise SystemExit(f"remote job failed (colab exec exit {rc}); see the log above")
        for name in [j["name"] for j in jobs] + ["remote_report"]:
            ext = ".json" if name == "remote_report" else ".mp4"
            self.c("download", "-s", self.s, f"{COLAB_REMOTE}/out/{name}{ext}", os.path.join(work, "out", name + ext),
                   check=False, timeout=900)

    def stop(self):
        if self.keep:
            log(f"leaving session {self.s} running (--keep): reuse it with --session {self.s}, "
                f"stop it with: colab --auth adc stop -s {self.s}")
        else:
            self.c("stop", "-s", self.s, check=False)


class Ssh:
    def __init__(self, args):
        if not args.host:
            raise SystemExit("--host user@machine is required for the ssh backend")
        self.host, self.py, self.rdir = args.host, args.remote_python, args.remote_dir
        if any(c.isspace() for c in self.rdir):
            raise SystemExit("--remote-dir must not contain spaces")
        self.qdir = shlex.quote(self.rdir) if not self.rdir.startswith("~/") else '"$HOME"/' + shlex.quote(self.rdir[2:])
        self.opts = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30"]
        if getattr(args, "ssh_key", None):
            self.opts += ["-i", os.path.expanduser(args.ssh_key), "-o", "IdentitiesOnly=yes"]
        self.port = str(getattr(args, "ssh_port", None) or 22)

    def ssh(self, cmd, **kw):
        return run(["ssh", *self.opts, "-p", self.port, self.host, cmd], **kw)

    def scp(self, *a, **kw):
        return run(["scp", "-q", *self.opts, "-P", self.port, *a], **kw)

    def check(self):
        r = self.ssh(f"{self.py} -c 'import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'",
                     check=False, timeout=120)
        if r.returncode or "True" not in r.stdout:
            return [f"{self.host}: CUDA PyTorch not usable with {self.py} ({(r.stderr or r.stdout).strip()[-300:]})"]
        log(f"{self.host}: {r.stdout.strip()}")
        return []

    def start(self):
        # out/ is emptied so a failed job can never hand back parts from an earlier run
        self.ssh(f"mkdir -p {self.qdir}/in {self.qdir}/out && rm -f {self.qdir}/out/* {self.qdir}/in/*")

    def run_job(self, work, jobs):
        files = [os.path.join(work, "in", f) for f in os.listdir(os.path.join(work, "in"))]
        self.scp(*files, f"{self.host}:{self.rdir}/in/")
        self.scp(*REMOTE_FILES, f"{self.host}:{self.rdir}/")
        rc = subprocess.run(["ssh", *self.opts, "-p", self.port, self.host,
                             f"cd {self.qdir} && {self.py} job.py --workdir ."]).returncode
        if rc:
            raise SystemExit(f"remote job failed on {self.host} (exit {rc}); see the log above")
        self.scp(f"{self.host}:{self.rdir}/out/*", os.path.join(work, "out") + "/", check=False)

    def stop(self):
        pass


class Runpod(Ssh):
    """A fresh RunPod GPU pod per run, driven by `runpodctl` and the API key it stores (no MCP, no browser login).

    Tries the GPUs in --runpod-gpu in order (stock changes minute to minute), waits for SSH, runs the same job as the
    ssh backend, and always deletes the pod at the end, also on errors and SIGTERM, unless --keep."""

    IMAGE = "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"  # CUDA torch, python 3.12, uv, git, ffmpeg, sshd

    def __init__(self, args):
        self.gpus = [g.strip() for g in args.runpod_gpu.split(",") if g.strip()]
        self.keep = getattr(args, "keep", False)
        self.cloud, self.disk = args.runpod_cloud, args.runpod_disk
        self.py, self.rdir, self.qdir = "python3", "/root/talkframe-work", "/root/talkframe-work"
        self.pod, self.cost_hr, self.t_start = None, None, None

    def rp(self, *a, **kw):
        return run(["runpodctl", *a], **kw)

    def balance(self):
        try:
            return float(json.loads(self.rp("user", check=False).stdout).get("clientBalance"))
        except Exception:
            return None

    def check(self):
        if not shutil.which("runpodctl"):
            return ["`runpodctl` not found; see references/backends.md (runpod)"]
        b = self.balance()
        if b is None:
            return ["runpodctl can't read the account; configure the key: runpodctl config --apiKey <key>"]
        log(f"runpod account ok, balance US${b:.2f}")
        return [] if b >= 1 else [f"low RunPod balance (US${b:.2f})"]

    def start(self):
        problems = self.check()
        if problems:
            raise SystemExit("runpod not usable: " + "; ".join(problems))
        name = f"talkframe-{time.strftime('%H%M%S')}"
        for gpu in self.gpus:
            log(f"runpod: creating a pod with {gpu}")
            r = self.rp("pod", "create", "--name", name, "--image", self.IMAGE, "--gpu-id", gpu,
                        "--cloud-type", self.cloud, "--container-disk-in-gb", str(self.disk), "--min-cuda-version", "12.8",
                        "--ports", "22/tcp", "--wait", "--wait-timeout", "12m", check=False, timeout=900)
            out = (r.stdout or "") + (r.stderr or "")
            try:
                d = json.loads(r.stdout)
            except ValueError:
                d = {}
            pid = d.get("id") or (re.search(r'"id"\s*:\s*"(\w+)"', out) or [None, None])[1]
            if r.returncode == 0 and d.get("ssh", {}).get("ip"):
                self.pod, self.cost_hr = d["id"], d.get("costPerHr")
                sshd = d["ssh"]
                self.host, self.port = f"root@{sshd['ip']}", str(sshd["port"])
                key = (sshd.get("ssh_key") or {}).get("path") or os.path.expanduser("~/.runpod/ssh/runpodctl-ssh-key")
                self.opts = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30",
                             "-o", "ConnectTimeout=15", "-i", key, "-o", "IdentitiesOnly=yes"]
                self.t_start = time.time()
                log(f"runpod: pod {self.pod} ({gpu}, US${self.cost_hr}/h, {d.get('machine', {}).get('dataCenterId')})")
                break
            if pid:  # created but never became reachable: don't leave it billing
                self.rp("pod", "delete", pid, check=False)
            if "no longer any instances" in out or "not available" in out.lower() or "timeout" in out.lower():
                log(f"runpod: no {gpu} available right now")
                continue
            raise SystemExit(f"runpod pod create failed: {out.strip()[-600:]}")
        else:
            log(f"runpod: none of {', '.join(self.gpus)} available; try later")
            sys.exit(EXIT_NO_GPU)
        # --wait only proves port 22 answers; the container may still be booting, so wait for a real login
        for i in range(60):
            if self.ssh("nvidia-smi -L", check=False, timeout=60).returncode == 0:
                break
            time.sleep(10)
        else:
            raise SystemExit(f"runpod: pod {self.pod} never accepted ssh")
        self.ssh(f"mkdir -p {self.qdir}/in {self.qdir}/out")

    def stop(self):
        if not self.pod:
            return
        if self.keep:
            log(f"leaving pod {self.pod} running (--keep); delete it with: runpodctl pod delete {self.pod}")
            return
        self.rp("pod", "delete", self.pod, check=False)
        log(f"runpod: pod {self.pod} deleted")

    def cost_usd(self):
        if self.cost_hr and self.t_start:
            return round(self.cost_hr * (time.time() - self.t_start) / 3600, 3)
        return None


class Local:
    def __init__(self, args):
        self.py = args.remote_python

    def check(self):
        r = run([self.py, "-c", "import torch;print(torch.cuda.is_available())"], check=False)
        return [] if "True" in r.stdout else ["no CUDA PyTorch on this machine; use --backend colab or ssh"]

    def start(self):
        pass

    def run_job(self, work, jobs):
        rc = subprocess.run([self.py, REMOTE_JOB, "--workdir", work]).returncode
        if rc:
            raise SystemExit(f"local job failed (exit {rc}); see the log above")

    def stop(self):
        pass


BACKENDS = {"colab": Colab, "ssh": Ssh, "runpod": Runpod, "local": Local}


# ---------------------------------------------------------------- stitching
def stitch(p, jobs, work):
    parts = [os.path.join(work, "out", j["name"] + ".mp4") for j in jobs]
    missing = [x for x in parts if not os.path.isfile(x)]
    if missing:
        raise SystemExit(f"backend did not return: {', '.join(map(os.path.basename, missing))} (see the log above)")
    # InfiniteTalk returns a few frames less than its audio (25 fps windows). Each part is padded with its last frame
    # and trimmed to exactly its audio segment's length, so later segments stay in sync and the end of the voice
    # isn't cut off. Then the untouched original audio goes over the joined picture (no seams in the sound).
    ins, chains = [], []
    for k, (x, j) in enumerate(zip(parts, jobs)):
        ins += ["-i", x]
        chains.append(f"[{k}:v]fps={FPS},tpad=stop_mode=clone:stop_duration=2,trim=end_frame={j['frames']},"
                      f"setpts=PTS-STARTPTS[v{k}]")
    graph = ";".join(chains) + ";" + "".join(f"[v{k}]" for k in range(len(parts))) + f"concat=n={len(parts)}:v=1:a=0[c]"
    # Optional lip-sync nudge: hold the first frame N frames to move the picture later. Default 0: on 2026-11-25 a
    # 4-frame default made the lips visibly late in a real reel, and onset measurements at 0 put the mouth ~1 frame
    # ahead of the voice, which is natural speech timing.
    graph += f";[c]tpad=start_mode=clone:start_duration={p.get('video_delay_frames', 0) / FPS}[v]"
    run(["ffmpeg", "-v", "error", "-y", *ins, "-i", p["audio"], "-filter_complex", graph,
         "-map", "[v]", "-map", f"{len(parts)}:a", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-t", str(p["audio_seconds_exact"]), "-movflags", "+faststart", p["output"]])


# ---------------------------------------------------------------- idle shutdown for --keep
LEASES = os.path.expanduser("~/.cache/talkframe/leases")


def lease_path(kind, name):
    return os.path.join(LEASES, f"{kind}-{name}.json")


def lease_update(kind, name, **fields):
    os.makedirs(LEASES, exist_ok=True)
    path = lease_path(kind, name)
    try:
        d = json.load(open(path))
    except (OSError, ValueError):
        d = {"kind": kind, "name": name}
    d.update(fields)
    tmp = path + ".tmp"
    json.dump(d, open(tmp, "w"))
    os.replace(tmp, path)
    return path


def lease_drop(kind, name):
    try:
        os.remove(lease_path(kind, name))
    except OSError:
        pass


def ensure_reaper(path):
    """Start a detached watcher that stops the VM/pod once its idle deadline passes (one watcher per lease)."""
    try:
        pid = json.load(open(path)).get("reaper_pid")
        if pid:
            os.kill(pid, 0)
            return  # already watching
    except (OSError, ValueError, TypeError):
        pass
    proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "_reap", path], start_new_session=True,
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    d = json.load(open(path))
    d["reaper_pid"] = proc.pid
    json.dump(d, open(path, "w"))


def reap(path):
    """Body of the detached watcher: wait while busy or before the deadline, then stop the session and remove the lease."""
    while True:
        try:
            d = json.load(open(path))
        except (OSError, ValueError):
            return  # lease gone: the VM was stopped normally
        if d.get("busy"):
            time.sleep(30)
            continue
        left = d.get("deadline", 0) - time.time()
        if left > 0:
            time.sleep(min(left, 60))
            continue
        if d["kind"] == "colab":
            subprocess.run(["colab", "--auth", "adc", "stop", "-s", d["name"]], capture_output=True)
        elif d["kind"] == "runpod":
            subprocess.run(["runpodctl", "pod", "delete", d["name"]], capture_output=True)
        try:
            os.remove(path)
        except OSError:
            pass
        return


def fingerprint(args):
    """Hash of everything that changes the rendered picture: same hash means the existing video is the same result."""
    import hashlib
    h = hashlib.sha256()
    for path in (args.image, args.audio):
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    for k in ("prompt", "resolution", "size", "steps", "seed", "audio_scale", "max_segment", "crop_y", "video_delay",
              "no_noise_floor", "protect_region", "attention", "fp8_fast", "quality", "model", "lc_mix_alpha",
              "lc_mix_steps"):
        h.update(f"{k}={getattr(args, k, None)}|".encode())
    return h.hexdigest()[:16]


def protect_python():
    """A Python that has numpy + OpenCV: this one if it does, else a throwaway env from `uv` (cached after first use)."""
    ok = subprocess.run([sys.executable, "-c", "import cv2, numpy"], capture_output=True).returncode == 0
    if ok:
        return [sys.executable]
    if shutil.which("uv"):
        return ["uv", "run", "--no-project", "-q", "--with", "numpy", "--with", "opencv-python-headless", "python"]
    raise SystemExit("--protect-region needs numpy+opencv: pip install numpy opencv-python-headless, install uv, "
                     "or pass --protect-python")


# ---------------------------------------------------------------- commands
def add_common(ap):
    ap.add_argument("--backend", choices=BACKENDS,
                    help="default: colab for infinitetalk, runpod for longcat (Colab's 40 GB A100 is too small for it)")
    ap.add_argument("--gpu", default="A100,H100", help="colab GPU or comma list of fallbacks: A100, H100, L4 (default A100,H100)")
    ap.add_argument("--gpu-attempts", type=int, default=10, help="colab: retries when no GPU is free (1/min)")
    ap.add_argument("--host", help="ssh backend: user@host")
    ap.add_argument("--remote-dir", default="talkframe-work", help="ssh backend: working dir on the host")
    ap.add_argument("--ssh-port", type=int, help="ssh backend: port (default 22)")
    ap.add_argument("--ssh-key", help="ssh backend: private key file")
    ap.add_argument("--runpod-gpu",
                    help="runpod: GPU ids to try in order. Default for infinitetalk: 48 GB+ cards (L40S, RTX 6000 Ada, "
                         "L40, A6000, A100 80GB); for longcat: 80 GB cards (A100 80GB, H100)")
    ap.add_argument("--runpod-cloud", choices=("SECURE", "COMMUNITY"), default="SECURE",
                    help="runpod: SECURE guarantees a public ssh port")
    ap.add_argument("--runpod-disk", type=int, default=120, help="runpod: container disk GB (models + ComfyUI ~60 GB)")
    ap.add_argument("--remote-python", default="python3", help="python with CUDA torch (ssh/local backends)")


def cmd_check(args):
    problems = []
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            problems.append(f"{tool} not found")
    problems += BACKENDS[args.backend](args).check()
    for x in problems:
        log(f"PROBLEM: {x}")
    log("check ok" if not problems else "check failed")
    return 1 if problems else 0


def cmd_generate(args):
    t0 = time.time()
    if args.protect_region and not args.dry_run:
        # resolve it before spending GPU time, so a missing OpenCV can't strand a finished render
        args.protect_cmd = [args.protect_python] if args.protect_python else protect_python()
    fp = fingerprint(args)
    rep_path = os.path.splitext(os.path.abspath(args.out))[0] + ".report.json"
    if not args.dry_run and not args.force and os.path.isfile(args.out) and os.path.isfile(rep_path):
        try:
            same = json.load(open(rep_path)).get("fingerprint") == fp
        except ValueError:
            same = False
        if same:
            # one render per reel: retries and later edits reuse this video instead of paying for the GPU again
            log(f"already rendered from these exact inputs: {args.out} (pass --force to render again)")
            return 0
    p, jobs, work = plan(args)
    p["fingerprint"] = fp
    log(f"plan: {p['audio_seconds']}s audio -> {len(jobs)} segment(s), {p['output_size'][0]}x{p['output_size'][1]}, "
        f"backend {args.backend}, workdir {work}")
    if args.dry_run:
        print(f"PLAN output_size={p['output_size'][0]}x{p['output_size'][1]} segments={p['segments']} model={args.model} steps={args.steps} attention={args.attention} fp8_fast={args.fp8_fast} photo_crop={','.join(map(str, p['photo_crop']))} "
              f"audio_seconds={p['audio_seconds']} backend={p['backend']} workdir={work}", flush=True)
        log(f"dry run: wrote {os.path.join(work, 'plan.json')} and in/jobs.json; nothing was rendered")
        print(json.dumps(p, indent=1))
        return 0
    be = BACKENDS[args.backend](args)
    # A queue that kills talkframe on a timeout sends SIGTERM: turn it into a normal exit so the `finally` below still
    # stops the Colab VM (otherwise it keeps billing until Colab reclaims it).
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    before = be.balance() if hasattr(be, "balance") else None
    out_dir = os.path.join(work, "out")
    shutil.rmtree(out_dir, ignore_errors=True)  # never stitch parts left over from an earlier run
    os.makedirs(out_dir)
    lease = None
    if isinstance(be, Colab) and be.reuse:
        lease = lease_update("colab", be.s, busy=True)  # pause any idle-shutdown timer while this job uses the VM
    try:
        be.start()
        be.run_job(work, jobs)
    finally:
        be.stop()
        name = be.s if isinstance(be, Colab) else getattr(be, "pod", None)
        kind = "colab" if isinstance(be, Colab) else "runpod" if isinstance(be, Runpod) else None
        if kind and name:
            if getattr(be, "keep", False) and args.keep_max > 0:
                lease = lease_update(kind, name, busy=False, deadline=time.time() + args.keep_max * 60)
                ensure_reaper(lease)
                log(f"{kind} {name} will be stopped automatically after {args.keep_max:g} idle minutes")
            elif not getattr(be, "keep", False):
                lease_drop(kind, name)
    stitch(p, jobs, work)
    if args.protect_region:
        # optional: paste the photo's exact pixels back over a flat detail (logo card, sign) that the model redraws
        raw = os.path.splitext(p["output"])[0] + ".raw.mp4"
        os.replace(p["output"], raw)
        # the region is given in the original photo; shift it into the cropped image the model actually saw
        rx, ry, rw, rh = (float(v) for v in args.protect_region.split(","))
        cx, cy = p["photo_crop"][:2]
        region = f"{rx - cx},{ry - cy},{rw},{rh}"
        model_img = [os.path.join(work, "in", f) for f in os.listdir(os.path.join(work, "in")) if f.startswith("image.")][0]
        r = subprocess.run([*args.protect_cmd, os.path.join(HERE, "protect.py"), "--video", raw, "--image", model_img,
                            "--region", region, "--out", p["output"]])
        if r.returncode:
            os.replace(raw, p["output"])
            log("protect-region failed; kept the unprotected video (see message above)")
        else:
            os.remove(raw)
    rr = os.path.join(work, "out", "remote_report.json")
    remote = json.load(open(rr)) if os.path.isfile(rr) else {}
    after = be.balance() if hasattr(be, "balance") else None
    used = round(before - after, 2) if before is not None and after is not None else None
    report = {**p, "wall_seconds": round(time.time() - t0), "gpu": remote.get("gpu"),
              "render_seconds": sum(j["render_seconds"] for j in remote.get("jobs", [])) or None,
              "colab_units_used": used if isinstance(be, Colab) else None,
              "runpod_usd_estimate": be.cost_usd() if isinstance(be, Runpod) else None}
    json.dump(report, open(rep_path, "w"), indent=1)
    cost = (f"units {report['colab_units_used']}" if isinstance(be, Colab) else
            f"~US${report['runpod_usd_estimate']}" if isinstance(be, Runpod) else "local")
    log(f"DONE {p['output']} ({report['wall_seconds']}s wall, {cost}) report: {rep_path}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="verify tools and backend access")
    add_common(c)
    g = sub.add_parser("generate", help="render a talking video")
    add_common(g)
    g.add_argument("--image", required=True, help="portrait photo (any orientation)")
    g.add_argument("--audio", required=True, help="voice track (wav/mp3/m4a...)")
    g.add_argument("--out", required=True, help="output .mp4 path")
    g.add_argument("--prompt", help="English scene description; see references/prompting.md")
    g.add_argument("--resolution", type=int, choices=(480, 720), default=480)
    g.add_argument("--size", help="exact WxH instead of --resolution (rounded down to multiples of 16)")
    g.add_argument("--model", choices=("infinitetalk", "longcat"), default="infinitetalk",
                   help="infinitetalk (default; Wan 2.1 + InfiniteTalk via ComfyUI, fits a 40 GB A100) or longcat "
                        "(LongCat-Video-Avatar 1.6: steadiest hands and identity in tests; needs an 80 GB GPU)")
    g.add_argument("--lc-mix-alpha", type=float, default=0.5,
                   help="longcat: DMD share in the distill LoRA mix (1.0 exaggerated the mouth; official default 0.5)")
    g.add_argument("--lc-mix-steps", type=int, default=3, help="longcat: steps that use the LoRA mix (official default 3)")
    g.add_argument("--quality", choices=("fast", "high"), default="high",
                   help="high (default): 6 steps, plain attention, fp8 storage; best lip-sync in tests (SyncNet LSE-C 7.98 "
                        "vs 7.02). fast: 4 steps + SageAttention + fp8 matmuls, ~35%% quicker but some syllables degrade. "
                        "--steps/--attention/--fp8-fast override it")
    g.add_argument("--steps", type=int, help="sampling steps (the lightx2v LoRA is distilled for 4-8)")
    g.add_argument("--attention", choices=("sdpa", "sageattn"), help="attention kernel (sageattn is installed on demand)")
    g.add_argument("--fp8-fast", choices=("auto", "on", "off"),
                   help="fp8 matmuls: auto uses them only on GPUs with fp8 hardware (Ada/Hopper/Blackwell, not A100)")
    g.add_argument("--force", action="store_true",
                   help="render again even if --out already holds a video made from the same inputs")
    g.add_argument("--seed", type=int, default=2)
    g.add_argument("--audio-scale", type=float, default=1.0, help="raise (1.2-1.5) for stronger lip motion")
    g.add_argument("--max-segment", type=float, default=60.0, help="split audio longer than this (seconds) at pauses")
    g.add_argument("--workdir", help="where inputs/parts are kept (default: <out>.talkframe/)")
    g.add_argument("--protect-region", help="X,Y,W,H in photo pixels of a flat logo/sign to keep pixel-exact (needs numpy+opencv)")
    g.add_argument("--protect-python", help="python with numpy+opencv for --protect-region (default: this one, else `uv run --with`)")
    g.add_argument("--crop-y", type=float, default=0.30,
                   help="when the video is wider than the photo: vertical centre of the kept band, as a fraction of the "
                        "photo's height (0.30 keeps the face in most portraits)")
    g.add_argument("--video-delay", type=int, default=0, help="frames to delay the picture vs the voice (lip-sync); 0 = none")
    g.add_argument("--no-noise-floor", action="store_true", help="send the audio to the model as-is (no noise floor in silences)")
    g.add_argument("--keep", action="store_true", help="colab/runpod: leave the VM running after the job (reuse with --session)")
    g.add_argument("--keep-max", type=float, default=15,
                   help="with --keep: stop the VM automatically after this many idle minutes (0 = never; default 15)")
    g.add_argument("--session", help="colab: run on this already-running session (e.g. one left by --keep); skips setup")
    g.add_argument("--dry-run", action="store_true", help="plan only: write plan.json/jobs.json, no GPU")
    args = ap.parse_args()
    model = getattr(args, "model", "infinitetalk")
    if args.backend is None:
        args.backend = "runpod" if model == "longcat" else "colab"
    if args.runpod_gpu is None:
        args.runpod_gpu = ("NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe,NVIDIA H100 80GB HBM3,NVIDIA H100 PCIe"
                           if model == "longcat" else
                           "NVIDIA L40S,NVIDIA RTX 6000 Ada Generation,NVIDIA L40,NVIDIA RTX A6000,"
                           "NVIDIA A100 80GB PCIe,NVIDIA A100-SXM4-80GB")
    if model == "longcat" and args.backend == "colab" and "H100" not in (args.gpu or ""):
        raise SystemExit("longcat needs an 80 GB GPU; Colab's A100 is 40 GB: use --backend runpod (default) or ssh")
    if args.cmd == "generate":
        preset = {"fast": (4, "sageattn", "auto"), "high": (6, "sdpa", "off")}[args.quality]
        args.steps = args.steps if args.steps is not None else preset[0]
        args.attention = args.attention or preset[1]
        args.fp8_fast = args.fp8_fast or preset[2]
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise SystemExit(f"{tool} is required")
    sys.exit(cmd_check(args) if args.cmd == "check" else cmd_generate(args))


if len(sys.argv) == 3 and sys.argv[1] == "_reap" and __name__ == "__main__":  # internal: detached idle watcher
    reap(sys.argv[2])
    sys.exit(0)


if __name__ == "__main__":
    main()
