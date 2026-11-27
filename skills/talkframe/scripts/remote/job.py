#!/usr/bin/env python3
"""talkframe remote job: runs ON the GPU machine (Colab VM or any Linux box with an NVIDIA GPU).

Steps (each is skipped when already done, so re-running on the same machine is cheap):
  1. install ComfyUI + kijai/ComfyUI-WanVideoWrapper + KJNodes + VideoHelperSuite with pinned deps
  2. download the InfiniteTalk model set (fp8, ~38 GB) from Hugging Face
  3. start a local ComfyUI server (127.0.0.1:8188)
  4. render every job in <workdir>/in/jobs.json to <workdir>/out/<name>.mp4

It prints a line at least every 30 s so `colab exec` never sees a silent kernel.
Only the Python standard library is needed to start; everything else is installed here.
"""
import argparse
import json
import secrets
import shlex
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

COMFY_PORT = 8188
NODE_REPOS = [
    "kijai/ComfyUI-WanVideoWrapper",
    "kijai/ComfyUI-KJNodes",
    "Kosinkadink/ComfyUI-VideoHelperSuite",
]
# transformers 5.x breaks kijai's wav2vec wrapper (no hidden_states); newer diffusers needs huggingface_hub>=1,
# which transformers 4.x does not accept. These pins were the working combination on 2026-11-24.
PINS = ["torchsde", "transformers==4.57.6", "diffusers==0.35.2"]
MODELS = [
    ("Kijai/WanVideo_comfy_fp8_scaled", "I2V/Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors", "diffusion_models/WanVideo"),
    ("Kijai/WanVideo_comfy_fp8_scaled", "InfiniteTalk/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors", "diffusion_models/WanVideo/InfiniteTalk"),
    ("Kijai/WanVideo_comfy", "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", "loras/WanVideo/Lightx2v"),
    ("Kijai/WanVideo_comfy", "umt5-xxl-enc-bf16.safetensors", "text_encoders"),
    ("Kijai/WanVideo_comfy", "Wan2_1_VAE_bf16.safetensors", "vae"),
    ("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/clip_vision/clip_vision_h.safetensors", "clip_vision"),
    ("Kijai/wav2vec2_safetensors", "wav2vec2-chinese-base_fp16.safetensors", "wav2vec2"),
]
NEGATIVE = (
    "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, "
    "worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, "
    "deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)


def log(msg):
    print(f"[talkframe {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sh(cmd, cwd=None):
    log(f"$ {cmd[:140]}")
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-4000:], flush=True)
        raise SystemExit(f"command failed: {cmd[:100]}")
    return r.stdout


CONSTRAINTS = None  # set by setup_comfy: pins the machine's own torch/torchvision/torchaudio


def pip_install(args):
    py = shlex.quote(sys.executable)
    base = f"uv pip install --python {py} -q" if shutil.which("uv") else f"{py} -m pip install -q"
    if CONSTRAINTS:
        base += f" -c {shlex.quote(CONSTRAINTS)}"
    r = subprocess.run(f"{base} {args}", shell=True, capture_output=True, text=True)
    if r.returncode and "externally managed" in (r.stdout + r.stderr).lower():
        # PEP 668 system Python (e.g. Ubuntu 24.04 images on RunPod): fine to override inside a throwaway GPU container
        log("system Python is marked externally managed; installing with --break-system-packages")
        sh(f"{base} --break-system-packages {args}")
    elif r.returncode:
        print(r.stdout[-2000:], r.stderr[-4000:], flush=True)
        raise SystemExit(f"pip install failed: {args[:100]}")


def setup_comfy(comfy):
    marker = f"{comfy}/.talkframe-setup-ok"
    if os.path.exists(marker):
        log("ComfyUI already installed")
        return
    if os.path.exists(comfy):
        log("found an unfinished ComfyUI install; starting it over")
        shutil.rmtree(comfy)
    q = shlex.quote
    # Keep the GPU machine's CUDA build of torch: without this, a loose requirement can upgrade torch to a newer
    # wheel while torchaudio/torchvision stay old, and torchaudio then fails with "undefined symbol" (seen on RunPod).
    global CONSTRAINTS
    import importlib.metadata as md
    pins = []
    for pkg in ("torch", "torchvision", "torchaudio"):
        try:
            pins.append(f"{pkg}=={md.version(pkg)}")
        except md.PackageNotFoundError:
            pass
    CONSTRAINTS = os.path.join(os.path.dirname(comfy), "torch-constraints.txt")
    os.makedirs(os.path.dirname(CONSTRAINTS), exist_ok=True)
    open(CONSTRAINTS, "w").write("\n".join(pins) + "\n")
    log("keeping " + ", ".join(pins))
    sh(f"git clone -q --depth 1 https://github.com/comfyanonymous/ComfyUI {q(comfy)}")
    for repo in NODE_REPOS:
        sh(f"git clone -q --depth 1 https://github.com/{repo} {q(comfy + '/custom_nodes/' + repo.split('/')[1])}")
    # never let ComfyUI's requirements replace the machine's CUDA build of torch
    sh(f"grep -v -E '^(torch|torchvision|torchaudio)([<>=~].*)?$' {q(comfy + '/requirements.txt')} "
       f"> {q(comfy + '/requirements.notorch.txt')}")
    reqs = " ".join(f"-r {q(comfy + '/custom_nodes/' + r.split('/')[1] + '/requirements.txt')}" for r in NODE_REPOS)
    pip_install(f"-r {q(comfy + '/requirements.notorch.txt')} {reqs}")
    pip_install(" ".join(q(p) for p in PINS))
    open(marker, "w").write(time.strftime("%Y-%m-%d %H:%M:%S\n"))  # only now is the install complete


def download_models(workdir):
    """Fetch the model set into <workdir>/models, several files at once. It runs in a thread while ComfyUI installs,
    so it lives outside the ComfyUI folder (an unfinished install gets wiped and redone)."""
    # public models: without this, huggingface_hub on Colab waits ~10 s for a notebook secret that never comes
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")  # faster Xet transfers (more parallel chunks)
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import hf_hub_download

    t0 = time.time()
    cache = os.path.join(workdir, "hf-download")

    def get(item):
        repo, path, sub = item
        dst = f"{workdir}/models/{sub}/{os.path.basename(path)}"
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.replace(hf_hub_download(repo, path, local_dir=cache), dst)
        return dst

    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(get, MODELS))
    log(f"models ready ({time.time() - t0:.0f}s)")


def link_models(comfy, workdir):
    """Expose <workdir>/models inside ComfyUI's models/ tree (symlinks, so nothing is copied)."""
    for repo, path, sub in MODELS:
        src = f"{workdir}/models/{sub}/{os.path.basename(path)}"
        dst = f"{comfy}/models/{sub}/{os.path.basename(path)}"
        if os.path.exists(dst):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.symlink(src, dst)


def _capture(errs, fn, *a):
    try:
        fn(*a)
    except BaseException as e:  # reported by the main thread
        errs.append(repr(e))


def _has(mod):
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


def comfy_up(port=None):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port or COMFY_PORT}/system_stats", timeout=3)
        return True
    except Exception:
        return False


def owns(comfy, port):
    """True if the server on `port` serves this install's input folder (another workdir's server must not be used)."""
    tok = secrets.token_hex(8)
    name = f".talkframe-owner-{tok}.txt"
    os.makedirs(f"{comfy}/input", exist_ok=True)
    with open(f"{comfy}/input/{name}", "w") as f:
        f.write(tok)
    try:
        got = urllib.request.urlopen(f"http://127.0.0.1:{port}/view?filename={name}&type=input", timeout=5).read().decode()
        return got.strip() == tok
    except Exception:
        return False
    finally:
        os.remove(f"{comfy}/input/{name}")


def start_comfy(comfy, workdir, restart=False):
    global COMFY_PORT
    pidfile = f"{workdir}/comfy.pid"
    if restart and comfy_up(8188):
        # a package the workflow needs (e.g. sageattention) is newer than our running server, which imported its nodes
        # without it: stop that server (ours: by pid file, or by its exact command line) and start a fresh one
        pids = []
        try:
            pids.append(int(open(pidfile).read()))
        except (OSError, ValueError):
            r = subprocess.run(["pgrep", "-f", "main.py --listen 127.0.0.1 --port 8188"], capture_output=True, text=True)
            pids = [int(x) for x in r.stdout.split()]
        for pid in pids:
            try:
                os.kill(pid, 15)
            except OSError:
                pass
        log("restarting ComfyUI so it picks up newly installed packages")
        for _ in range(30):
            if not comfy_up(8188):
                break
            time.sleep(1)
    for port in range(8188, 8198):
        if not comfy_up(port):
            COMFY_PORT = port
            break
        if owns(comfy, port):
            COMFY_PORT = port
            log(f"ComfyUI already running on :{port}")
            return
        log(f"port {port} is another ComfyUI (different install); trying the next port")
    else:
        raise SystemExit("no free port in 8188-8197 for ComfyUI")
    logf = open(f"{workdir}/comfy.log", "a")
    proc = subprocess.Popen([sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(COMFY_PORT),
                             "--disable-auto-launch"], cwd=comfy, stdout=logf, stderr=subprocess.STDOUT)
    open(f"{workdir}/comfy.pid", "w").write(str(proc.pid))
    for i in range(200):
        if comfy_up():
            break
        if i % 10 == 0:
            log("waiting for ComfyUI...")
        time.sleep(3)
    else:
        print(open(f"{workdir}/comfy.log").read()[-4000:])
        raise SystemExit("ComfyUI did not start")
    info = json.load(urllib.request.urlopen(f"http://127.0.0.1:{COMFY_PORT}/object_info/WanVideoSampler"))
    if not info:
        print(open(f"{workdir}/comfy.log").read()[-4000:])
        raise SystemExit("ComfyUI-WanVideoWrapper failed to import (see comfy.log)")
    log("ComfyUI up")


def quantization(job):
    """fp8 matmuls need fp8 tensor cores (compute capability 8.9+: Ada, Hopper, Blackwell). A100 (8.0) keeps fp8 storage only."""
    mode = job.get("fp8_fast", "off")
    if mode == "auto":
        import torch
        mode = "on" if torch.cuda.get_device_capability(0) >= (8, 9) else "off"
    return "fp8_e4m3fn_scaled_fast" if mode == "on" else "fp8_e4m3fn_scaled"


def build_graph(job, image_name, audio_name):
    w, h = job["width"], job["height"]
    g = {
        "lora": {"class_type": "WanVideoLoraSelect", "inputs": {
            "lora": "WanVideo/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", "strength": 1.0}},
        "mt": {"class_type": "MultiTalkModelLoader", "inputs": {
            "model": "WanVideo/InfiniteTalk/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors"}},
        "model": {"class_type": "WanVideoModelLoader", "inputs": {
            "model": "WanVideo/Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors", "base_precision": "bf16",
            "quantization": quantization(job), "load_device": "main_device",
            "attention_mode": job.get("attention", "sdpa"),
            "lora": ["lora", 0], "multitalk_model": ["mt", 0]}},
        "vae": {"class_type": "WanVideoVAELoader", "inputs": {"model_name": "Wan2_1_VAE_bf16.safetensors", "precision": "bf16"}},
        "clipv": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "clip_vision_h.safetensors"}},
        "img": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "resize": {"class_type": "ImageResizeKJv2", "inputs": {
            "image": ["img", 0], "width": w, "height": h, "upscale_method": "bicubic",
            "keep_proportion": "crop", "pad_color": "0, 0, 0", "crop_position": "center", "divisible_by": 16, "device": "cpu"}},
        "clipenc": {"class_type": "WanVideoClipVisionEncode", "inputs": {
            "clip_vision": ["clipv", 0], "image_1": ["resize", 0], "strength_1": 1.0, "strength_2": 1.0, "crop": "center",
            "combine_embeds": "average", "force_offload": True, "tiles": 0, "ratio": 0.5}},
        "aud": {"class_type": "LoadAudio", "inputs": {"audio": audio_name}},
        "w2v": {"class_type": "Wav2VecModelLoader", "inputs": {
            "model": "wav2vec2-chinese-base_fp16.safetensors", "base_precision": "fp16", "load_device": "main_device"}},
        "emb": {"class_type": "MultiTalkWav2VecEmbeds", "inputs": {
            "wav2vec_model": ["w2v", 0], "audio_1": ["aud", 0], "normalize_loudness": True, "num_frames": 2000, "fps": 25.0,
            "audio_scale": job.get("audio_scale", 1.0), "audio_cfg_scale": 1.0, "multi_audio_type": "para"}},
        "text": {"class_type": "WanVideoTextEncodeCached", "inputs": {
            "model_name": "umt5-xxl-enc-bf16.safetensors", "precision": "bf16", "positive_prompt": job["prompt"],
            "negative_prompt": NEGATIVE, "quantization": "disabled", "use_disk_cache": False, "device": "gpu"}},
        "i2v": {"class_type": "WanVideoImageToVideoMultiTalk", "inputs": {
            "vae": ["vae", 0], "width": ["resize", 1], "height": ["resize", 2], "frame_window_size": 81, "motion_frame": 9,
            "force_offload": False, "colormatch": "disabled", "start_image": ["resize", 0], "tiled_vae": False,
            "clip_embeds": ["clipenc", 0], "mode": "infinitetalk", "output_path": ""}},
        "sampler": {"class_type": "WanVideoSampler", "inputs": {
            "model": ["model", 0], "image_embeds": ["i2v", 0], "text_embeds": ["text", 0], "multitalk_embeds": ["emb", 0],
            "steps": job.get("steps", 6), "cfg": 1.0, "shift": 11.0, "seed": job.get("seed", 2), "force_offload": True,
            "scheduler": "dpm++_sde", "riflex_freq_index": 0, "rope_function": "comfy"}},
        "frames": {"class_type": "WanVideoPassImagesFromSamples", "inputs": {"samples": ["sampler", 0]}},
        "out": {"class_type": "VHS_VideoCombine", "inputs": {
            "images": ["frames", 0], "audio": ["aud", 0], "frame_rate": 25, "loop_count": 0, "filename_prefix": job["name"],
            "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 17, "save_metadata": False, "trim_to_audio": True,
            "pingpong": False, "save_output": True}},
    }
    if job.get("negative"):
        # At cfg 1 (lightx2v LoRA) the regular negative prompt is never evaluated. NAG (Normalized Attention Guidance)
        # applies it inside cross-attention instead, at no extra sampling pass: the encoder's 2nd output holds only it.
        g["text"]["inputs"]["negative_prompt"] = job["negative"]
        g["nag"] = {"class_type": "WanVideoApplyNAG", "inputs": {
            "original_text_embeds": ["text", 0], "nag_text_embeds": ["text", 1],
            "nag_scale": job.get("nag_scale", 11.0), "nag_tau": 2.5, "nag_alpha": 0.25}}
        g["sampler"]["inputs"]["text_embeds"] = ["nag", 0]
    return g


def api(path, payload=None):
    url = f"http://127.0.0.1:{COMFY_PORT}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))


def render(job, comfy, workdir):
    t0 = time.time()
    inp = f"{comfy}/input"
    os.makedirs(inp, exist_ok=True)
    image_name = f"{job['name']}_{os.path.basename(job['image'])}"
    audio_name = f"{job['name']}_{os.path.basename(job['audio'])}"
    shutil.copy(f"{workdir}/in/{job['image']}", f"{inp}/{image_name}")
    shutil.copy(f"{workdir}/in/{job['audio']}", f"{inp}/{audio_name}")
    try:
        pid = api("/prompt", {"prompt": build_graph(job, image_name, audio_name)})["prompt_id"]
    except urllib.error.HTTPError as e:
        raise SystemExit(f"ComfyUI rejected the workflow: {e.read().decode()[:3000]}")
    log(f"{job['name']}: queued")
    last = 0
    while True:
        hist = api(f"/history/{pid}")
        if pid in hist:
            status = hist[pid]["status"]
            if status.get("status_str") != "success":
                raise SystemExit(f"{job['name']} failed: {json.dumps(status)[:3000]}")
            break
        if time.time() - last > 25:
            log(f"{job['name']}: rendering ({time.time() - t0:.0f}s)")
            last = time.time()
        time.sleep(5)
    outs = [os.path.join(comfy, "output", f) for f in os.listdir(f"{comfy}/output") if f.startswith(job["name"]) and f.endswith(".mp4")]
    best = max(outs, key=os.path.getmtime)
    shutil.copy(best, f"{workdir}/out/{job['name']}.mp4")
    secs = time.time() - t0
    log(f"{job['name']}: done in {secs:.0f}s")
    return secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=os.environ.get("TALKFRAME_WORKDIR", "/content/talkframe"))
    a, _ = ap.parse_known_args()  # parse_known_args: under `colab exec` sys.argv holds the kernel's own flags
    workdir = os.path.abspath(a.workdir)
    comfy = f"{workdir}/ComfyUI"
    os.makedirs(f"{workdir}/out", exist_ok=True)

    import importlib.util
    if importlib.util.find_spec("torch") is None:
        raise SystemExit("PyTorch with CUDA must be preinstalled on the GPU machine (Colab already has it).")
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU visible to PyTorch.")
    log(f"GPU: {torch.cuda.get_device_name(0)}, torch {torch.__version__}")

    jobs = json.load(open(f"{workdir}/in/jobs.json"))
    if not _has("huggingface_hub"):  # both model paths download weights from the machine's own python
        pip_install("huggingface_hub")
    if any(j.get("model") == "longcat" for j in jobs):
        # longcat.py ships next to job.py: in the work dir (Colab/ssh) or in scripts/remote (local backend).
        # Under `colab exec` this code runs inside the kernel, where __file__ is not defined.
        here = os.path.dirname(os.path.abspath(globals().get("__file__", f"{workdir}/job.py")))
        for d in (workdir, here):
            if d not in sys.path:
                sys.path.insert(0, d)
        sys.modules.pop("longcat", None)  # a reused Colab kernel may hold an older copy of the module
        import longcat
        report = longcat.run(workdir, jobs)
        json.dump({"gpu": torch.cuda.get_device_name(0), "model": "longcat", "jobs": report},
                  open(f"{workdir}/out/remote_report.json", "w"), indent=1)
        log("ALL_DONE")
        return
    # download the ~38 GB of models while ComfyUI and its dependencies install (they don't depend on each other)
    if not _has("huggingface_hub"):
        pip_install("huggingface_hub")
    import threading
    dl_err = []
    dl = threading.Thread(target=lambda: _capture(dl_err, download_models, workdir), daemon=True)
    dl.start()
    setup_comfy(comfy)
    extra = {j.get("attention") for j in json.load(open(f"{workdir}/in/jobs.json"))}
    restart = False
    if "sageattn" in extra:
        if not _has("sageattention"):
            pip_install("sageattention==1.0.6")
        import importlib.util
        pkg = os.path.dirname(importlib.util.find_spec("sageattention").origin)
        pidfile = f"{workdir}/comfy.pid"
        started = os.path.getmtime(pidfile) if os.path.exists(pidfile) else 0
        restart = os.path.getmtime(pkg) > started  # server older than the package (or unknown): restart it
    while dl.is_alive():
        log("still downloading models...")
        dl.join(25)
    if dl_err:
        raise SystemExit(f"model download failed: {dl_err[0]}")
    link_models(comfy, workdir)
    start_comfy(comfy, workdir, restart=restart)
    report = []
    for job in jobs:
        report.append({"name": job["name"], "render_seconds": round(render(job, comfy, workdir), 1)})
    json.dump({"gpu": torch.cuda.get_device_name(0), "jobs": report}, open(f"{workdir}/out/remote_report.json", "w"), indent=1)
    log("ALL_DONE")


if __name__ == "__main__":
    main()
