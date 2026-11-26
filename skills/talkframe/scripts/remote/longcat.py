"""LongCat-Video-Avatar 1.6 (Meituan, MIT) for talkframe: install + render, run on the GPU machine by job.py.

Recipe proven on RunPod (runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404, A100/H100 80 GB, 2026-11-26):
- its own venv (--system-site-packages) so the machine's CUDA torch is reused and pinned by a constraints file;
- prebuilt flash-attn wheel (the repo has no SDPA fallback); pinned transformers/diffusers;
- ~60 GB of weights downloaded while the environment installs;
- distilled 8-step mode with the official 50/50 DMD/OPSD LoRA mix (mix_steps 3, alpha 0.5): 100% DMD made the
  mouth look exaggerated; non-distilled 30 steps is ~7x slower for a small gain;
- OMP/MKL threads capped at 16: containers report every host core and torch otherwise stalls for minutes loading.
Needs ~62 GB of VRAM at 480p (bf16): an 80 GB GPU.
"""
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time

REPO = "https://github.com/meituan-longcat/LongCat-Video"
COMMIT = "58eb336edc"
WEIGHTS = [
    ("meituan-longcat/LongCat-Video", "LongCat-Video",
     ["tokenizer/*", "text_encoder/*", "vae/*", "scheduler/*", "config.json", "model_index.json"]),
    ("meituan-longcat/LongCat-Video-Avatar-1.6", "LongCat-Video-Avatar-1.6",
     ["base_model/*", "loraA/*", "loraB/*", "scheduler/*", "vocal_separator/*", "config.json", "model_index.json",
      "whisper-large-v3/*.json", "whisper-large-v3/merges.txt", "whisper-large-v3/model.safetensors"]),
]
PIP = ["numpy==1.26.4", "transformers==4.41.0", "huggingface_hub>=0.34,<1.0", "diffusers==0.35.1", "loguru==0.7.2",
       "einops==0.8.0", "ftfy==6.2.0", "imageio==2.37.0", "imageio-ffmpeg==0.6.0", "safetensors", "regex", "tqdm",
       "sentencepiece", "protobuf", "scipy==1.15.3", "soundfile==0.13.1", "soxr==0.5.0.post1", "librosa==0.11.0",
       "pyloudnorm==0.1.1", "onnx==1.18.0", "onnxruntime-gpu==1.20.2", "audio-separator==0.30.2"]
# prebuilt flash-attn wheels by (torch minor, python tag); building from source takes ~1 h
FLASH = {("2.9", "cp312"): "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
                           "flash_attn-2.8.3%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"}


def log(msg):
    print(f"[talkframe {time.strftime('%H:%M:%S')}] longcat: {msg}", flush=True)


def sh(cmd, env=None, cwd=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env, cwd=cwd)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-4000:], flush=True)
        raise SystemExit(f"longcat setup failed: {cmd[:120]}")
    return r.stdout


def download(workdir, errs):
    try:
        os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
        from huggingface_hub import snapshot_download
        t0 = time.time()
        for repo, sub, pats in WEIGHTS:
            snapshot_download(repo, local_dir=f"{workdir}/lc-weights/{sub}", allow_patterns=pats, max_workers=8)
        log(f"weights ready ({time.time() - t0:.0f}s)")
    except BaseException as e:
        errs.append(repr(e))


def setup(workdir):
    import torch
    venv, repo = f"{workdir}/lcenv", f"{workdir}/LongCat-Video"
    marker = f"{venv}/.talkframe-longcat-ok"
    if os.path.exists(marker):
        log("environment already installed")
        return venv, repo
    for d in (venv, repo):
        shutil.rmtree(d, ignore_errors=True)
    if not shutil.which("ffmpeg"):
        sh("apt-get update -qq && apt-get install -y -qq ffmpeg libsndfile1 libsamplerate0 >/dev/null")
    sh(f"{sys.executable} -m venv --system-site-packages {venv}")
    py = f"{venv}/bin/python"
    sh(f"{py} -m pip install -q -U pip wheel setuptools")
    cons = f"{workdir}/lc-constraints.txt"
    sh(f"{py} -m pip freeze | grep -iE '^(torch|torchvision|torchaudio|triton)==' > {cons}; true")
    if subprocess.run([py, "-c", "import torchvision"], capture_output=True).returncode:
        tv = {"2.9": "0.24.1", "2.8": "0.23.0"}.get(".".join(torch.__version__.split(".")[:2]))
        if tv:
            sh(f"{py} -m pip install -q torchvision=={tv} --index-url https://download.pytorch.org/whl/cu128 -c {cons}")
    key = (".".join(torch.__version__.split(".")[:2]), f"cp{sys.version_info.major}{sys.version_info.minor}")
    wheel = FLASH.get(key)
    if not wheel:
        raise SystemExit(f"no prebuilt flash-attn wheel for torch {key[0]} / {key[1]}; LongCat was tested on "
                         "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 (use --backend runpod)")
    sh(f"{py} -m pip install -q ninja packaging psutil")
    sh(f"{py} -m pip install -q -c {cons} '{wheel}'")
    sh(f"{py} -m pip install -q -c {cons} " + " ".join(f"'{p}'" for p in PIP))
    sh(f"git clone -q --depth 50 --single-branch --branch main {REPO} {repo} && cd {repo} && git checkout -q {COMMIT}")
    sh(f"{py} -c 'import flash_attn, transformers, diffusers, audio_separator.separator, librosa'")
    open(marker, "w").write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    log("environment installed")
    return venv, repo


def render(job, workdir, venv, repo):
    t0 = time.time()
    img, aud = f"{workdir}/in/{job['image']}", f"{workdir}/in/{job['audio']}"
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", aud],
                               capture_output=True, text=True).stdout or 0)
    # LongCat renders 93 frames, then 80 new frames per extra segment, at 25 fps
    segs = 1 + max(0, math.ceil((dur * 25 - 93) / 80))
    jdir = f"{workdir}/lc-jobs/{job['name']}"
    shutil.rmtree(jdir, ignore_errors=True)
    os.makedirs(f"{jdir}/out")
    json.dump({"prompt": job["prompt"], "cond_image": img, "cond_audio": {"person1": aud}}, open(f"{jdir}/job.json", "w"))
    threads = str(min(16, os.cpu_count() or 16))
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "OMP_NUM_THREADS": threads,
           "MKL_NUM_THREADS": threads, "PATH": f"{venv}/bin:" + os.environ.get("PATH", "")}
    args = [f"{venv}/bin/python", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1",
            "run_demo_avatar_single_audio_to_video.py", "--context_parallel_size=1",
            f"--checkpoint_dir={workdir}/lc-weights/LongCat-Video-Avatar-1.6", "--stage_1=ai2v",
            "--model_type", "avatar-v1.6", f"--input_json={jdir}/job.json",
            f"--resolution={'720p' if job.get('resolution') == 720 else '480p'}", f"--num_segments={segs}",
            "--ref_img_index=10", "--mask_frame_range=0", "--use_distill", "--lora_path", "loraB/lora.safetensors",
            f"--mix_steps={job.get('lc_mix_steps', 3)}", f"--mix_alpha={job.get('lc_mix_alpha', 0.5)}",
            f"--output_dir={jdir}/out"]
    log(f"{job['name']}: {dur:.1f}s audio -> {segs} segment(s)")
    with open(f"{jdir}/gen.log", "w") as lf:
        p = subprocess.Popen(args, cwd=repo, env=env, stdout=lf, stderr=subprocess.STDOUT)
        last = time.time()
        while p.poll() is None:
            time.sleep(5)
            if time.time() - last > 25:
                log(f"{job['name']}: rendering ({time.time() - t0:.0f}s)")
                last = time.time()
    if p.returncode:
        print(open(f"{jdir}/gen.log").read()[-4000:], flush=True)
        raise SystemExit(f"longcat render failed for {job['name']}")
    outs = sorted((os.path.join(f"{jdir}/out", f) for f in os.listdir(f"{jdir}/out") if f.endswith(".mp4")),
                  key=os.path.getmtime)
    if not outs:
        raise SystemExit(f"longcat produced no video for {job['name']}")
    shutil.copy(outs[-1], f"{workdir}/out/{job['name']}.mp4")
    secs = time.time() - t0
    log(f"{job['name']}: done in {secs:.0f}s")
    return secs


def run(workdir, jobs):
    errs = []
    dl = threading.Thread(target=download, args=(workdir, errs), daemon=True)
    dl.start()
    venv, repo = setup(workdir)
    while dl.is_alive():
        log("still downloading weights...")
        dl.join(25)
    if errs:
        raise SystemExit(f"longcat weight download failed: {errs[0]}")
    return [{"name": j["name"], "render_seconds": round(render(j, workdir, venv, repo), 1)} for j in jobs]
