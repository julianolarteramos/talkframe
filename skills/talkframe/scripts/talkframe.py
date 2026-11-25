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
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REMOTE_JOB = os.path.join(HERE, "remote", "job.py")
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

    ext = os.path.splitext(args.image)[1].lower() or ".png"
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
                     # whole frames this part must fill, from absolute frame positions so rounding never accumulates
                     "frames": round(bounds[i + 1] * FPS) - round(bounds[i] * FPS)})
    json.dump(jobs, open(os.path.join(work, "in", "jobs.json"), "w"), indent=1)
    p = {"image": os.path.abspath(args.image), "image_size": [iw, ih], "audio": os.path.abspath(args.audio),
         "audio_seconds": round(dur, 2), "audio_seconds_exact": dur, "output": out, "output_size": [w, h], "segments": len(jobs),
         "video_delay_frames": args.video_delay, "cut_points": cuts, "dry_run": bool(args.dry_run), "backend": args.backend, "workdir": work, "prompt": args.prompt or DEFAULT_PROMPT}
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
        for i in range(self.attempts):
            gpu = gpus[i % len(gpus)]
            try:
                r = self.c("new", "-s", self.s, "--gpu", gpu, check=False, timeout=300)
                err = (r.stdout + r.stderr).lower()
            except subprocess.TimeoutExpired:
                err = "timeout"
            if self.alive():
                log(f"colab session {self.s} ({gpu})")
                return
            if not any(k in err for k in ("unavailable", "503", "timeout", "capacity", "try again")):
                raise SystemExit(f"colab new failed (not a busy pool): {err.strip()[-800:]}")
            if i + 1 < self.attempts:
                log(f"no {gpu} available, attempt {i + 1}/{self.attempts}; retrying in 60s")
                time.sleep(60)
        self.c("stop", "-s", self.s, check=False)  # a timed-out `new` may still come up later; don't leave it billing
        log(f"could not get a Colab GPU ({self.gpu}) after {self.attempts} attempts; try later")
        sys.exit(EXIT_NO_GPU)

    def run_job(self, work, jobs):
        p = subprocess.run(["colab", "--auth", "adc", "exec", "-s", self.s],
                           input=(f"import os, glob\nfor d in ('in', 'out'):\n    os.makedirs('{COLAB_REMOTE}/' + d, exist_ok=True)\n"
                                  f"    [os.remove(f) for f in glob.glob('{COLAB_REMOTE}/' + d + '/*') if os.path.isfile(f)]\n"),
                           text=True, capture_output=True)
        if p.returncode:
            raise SystemExit(f"colab exec failed: {p.stderr[-1500:]}")
        files = [os.path.join(work, "in", f) for f in os.listdir(os.path.join(work, "in"))]
        for f in files:
            self.c("upload", "-s", self.s, f, f"{COLAB_REMOTE}/in/{os.path.basename(f)}", timeout=900)
        self.c("upload", "-s", self.s, REMOTE_JOB, f"{COLAB_REMOTE}/job.py", timeout=300)
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

    def ssh(self, cmd, **kw):
        return run(["ssh", "-o", "BatchMode=yes", self.host, cmd], **kw)

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
        run(["scp", "-q", *files, f"{self.host}:{self.rdir}/in/"])
        run(["scp", "-q", REMOTE_JOB, f"{self.host}:{self.rdir}/job.py"])
        rc = subprocess.run(["ssh", "-o", "BatchMode=yes", self.host,
                             f"cd {self.qdir} && {self.py} job.py --workdir ."]).returncode
        if rc:
            raise SystemExit(f"remote job failed on {self.host} (exit {rc}); see the log above")
        run(["scp", "-q", f"{self.host}:{self.rdir}/out/*", os.path.join(work, "out") + "/"], check=False)

    def stop(self):
        pass


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


BACKENDS = {"colab": Colab, "ssh": Ssh, "local": Local}


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
    # InfiniteTalk's mouth leads the voice by ~0.1-0.2 s (it anticipates the next word). Holding the first frame for
    # a few frames shifts the picture later; 4 frames (160 ms) looked best in side-by-side tests.
    graph += f";[c]tpad=start_mode=clone:start_duration={p.get('video_delay_frames', 4) / FPS}[v]"
    run(["ffmpeg", "-v", "error", "-y", *ins, "-i", p["audio"], "-filter_complex", graph,
         "-map", "[v]", "-map", f"{len(parts)}:a", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-t", str(p["audio_seconds_exact"]), "-movflags", "+faststart", p["output"]])


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
    ap.add_argument("--backend", choices=BACKENDS, default="colab")
    ap.add_argument("--gpu", default="A100,H100", help="colab GPU or comma list of fallbacks: A100, H100, L4 (default A100,H100)")
    ap.add_argument("--gpu-attempts", type=int, default=10, help="colab: retries when no GPU is free (1/min)")
    ap.add_argument("--host", help="ssh backend: user@host")
    ap.add_argument("--remote-dir", default="talkframe-work", help="ssh backend: working dir on the host")
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
    p, jobs, work = plan(args)
    log(f"plan: {p['audio_seconds']}s audio -> {len(jobs)} segment(s), {p['output_size'][0]}x{p['output_size'][1]}, "
        f"backend {args.backend}, workdir {work}")
    if args.dry_run:
        print(f"PLAN output_size={p['output_size'][0]}x{p['output_size'][1]} segments={p['segments']} "
              f"audio_seconds={p['audio_seconds']} backend={p['backend']} workdir={work}", flush=True)
        log(f"dry run: wrote {os.path.join(work, 'plan.json')} and in/jobs.json; nothing was rendered")
        print(json.dumps(p, indent=1))
        return 0
    be = BACKENDS[args.backend](args)
    before = be.balance() if isinstance(be, Colab) else None
    out_dir = os.path.join(work, "out")
    shutil.rmtree(out_dir, ignore_errors=True)  # never stitch parts left over from an earlier run
    os.makedirs(out_dir)
    try:
        be.start()
        be.run_job(work, jobs)
    finally:
        be.stop()
    stitch(p, jobs, work)
    if args.protect_region:
        # optional: paste the photo's exact pixels back over a flat detail (logo card, sign) that the model redraws
        raw = os.path.splitext(p["output"])[0] + ".raw.mp4"
        os.replace(p["output"], raw)
        r = subprocess.run([*args.protect_cmd, os.path.join(HERE, "protect.py"), "--video", raw, "--image", p["image"],
                            "--region", args.protect_region, "--out", p["output"]])
        if r.returncode:
            os.replace(raw, p["output"])
            log("protect-region failed; kept the unprotected video (see message above)")
        else:
            os.remove(raw)
    rr = os.path.join(work, "out", "remote_report.json")
    remote = json.load(open(rr)) if os.path.isfile(rr) else {}
    after = be.balance() if isinstance(be, Colab) else None
    report = {**p, "wall_seconds": round(time.time() - t0), "gpu": remote.get("gpu"),
              "render_seconds": sum(j["render_seconds"] for j in remote.get("jobs", [])) or None,
              "colab_units_used": round(before - after, 2) if before is not None and after is not None else None}
    rep_path = os.path.splitext(p["output"])[0] + ".report.json"
    json.dump(report, open(rep_path, "w"), indent=1)
    log(f"DONE {p['output']} ({report['wall_seconds']}s wall, units {report['colab_units_used']}) report: {rep_path}")
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
    g.add_argument("--steps", type=int, default=6)
    g.add_argument("--seed", type=int, default=2)
    g.add_argument("--audio-scale", type=float, default=1.0, help="raise (1.2-1.5) for stronger lip motion")
    g.add_argument("--max-segment", type=float, default=60.0, help="split audio longer than this (seconds) at pauses")
    g.add_argument("--workdir", help="where inputs/parts are kept (default: <out>.talkframe/)")
    g.add_argument("--protect-region", help="X,Y,W,H in photo pixels of a flat logo/sign to keep pixel-exact (needs numpy+opencv)")
    g.add_argument("--protect-python", help="python with numpy+opencv for --protect-region (default: this one, else `uv run --with`)")
    g.add_argument("--video-delay", type=int, default=4, help="frames to delay the picture vs the voice (lip-sync); 0 = none")
    g.add_argument("--no-noise-floor", action="store_true", help="send the audio to the model as-is (no noise floor in silences)")
    g.add_argument("--keep", action="store_true", help="colab: leave the VM running after the job (reuse with --session)")
    g.add_argument("--session", help="colab: run on this already-running session (e.g. one left by --keep); skips setup")
    g.add_argument("--dry-run", action="store_true", help="plan only: write plan.json/jobs.json, no GPU")
    args = ap.parse_args()
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise SystemExit(f"{tool} is required")
    sys.exit(cmd_check(args) if args.cmd == "check" else cmd_generate(args))


if __name__ == "__main__":
    main()
