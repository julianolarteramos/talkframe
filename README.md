# talkframe

**Make a photo talk.** Give it a portrait and a voice recording, and talkframe returns a video where that person speaks the audio with natural facial expressions, head and hand movement, moving hair and a living background, not just flapping lips.

It is a [Claude Code](https://code.claude.com) plugin with one skill. Ask Claude something like *"make this photo of our presenter say narration.mp3, as a horizontal strip for the bottom of a reel"*. Claude writes the scene prompt from the photo, rents a GPU, renders, stitches and checks the result. Nobody has to babysit a notebook.

Under the hood:
- **[InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk)** for audio-driven motion, on top of **Wan 2.1 I2V 14B**.
- Run through **ComfyUI** with **[kijai/ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)**.
- fp8 weights and the lightx2v step-distill LoRA (6 sampling steps).

## Install

```text
/plugin marketplace add julianolarteramos/talkframe
/plugin install talkframe@talkframe
```

Locally you need Python 3.9+ and `ffmpeg`. The GPU side installs itself.

## Pick a GPU backend

| Backend | What you need | Cost and speed (measured) |
|---|---|---|
| `colab` (default) | Colab Pro or pay-as-you-go compute units, plus the [Colab CLI](https://github.com/googlecolab/google-colab-cli) | A100 at ~8 units/h. A 30 s clip takes ~15 min including setup, ~2 units (~US$0.20). |
| `ssh` | Any Linux host with an NVIDIA GPU (≥24 GB), CUDA PyTorch and key-based SSH | Your machine's or cloud provider's rate |
| `local` | This machine has the NVIDIA GPU | Free |

One-time Colab setup:

```bash
uv tool install google-colab-cli
gcloud auth application-default login --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
```

Details and pitfalls are in [`skills/talkframe/references/backends.md`](skills/talkframe/references/backends.md).

## Use it without Claude

```bash
S=skills/talkframe/scripts/talkframe.py
python3 $S check --backend colab
python3 $S generate --image host.jpg --audio narration.mp3 --out out/host.mp4 \
  --prompt "A news presenter at a desk speaks to the camera, nodding and gesturing with her hands..." \
  --resolution 480            # or 720, or --size 1024x576
python3 $S generate ... --dry-run   # plan only, no GPU
```

What `generate` does:
- The output keeps the photo's aspect ratio at ~0.4 MP (480) or ~0.9 MP (720).
- Audio longer than 60 s is split at pauses, rendered in parts and joined under the original soundtrack.
- `<out>.report.json` records the size, segments, GPU, wall time and Colab units used.

## How it works

```
your machine                                  GPU machine (Colab VM / SSH host / local)
─────────────                                  ────────────────────────────────────────
talkframe.py generate                          job.py
  ├─ probe image + audio                         ├─ install ComfyUI + nodes (pinned deps)
  ├─ pick size, split audio at pauses            ├─ download ~38 GB fp8 models from Hugging Face
  ├─ write jobs.json ─────── upload ───────────► ├─ start ComfyUI headless
  ├─ start / stop the backend                    └─ render each segment via the ComfyUI API
  └─ stitch parts + original audio ◄── download ── out/segNNN.mp4
```

## Tests

`evals/` holds a suite for [`claude plugin eval`](https://code.claude.com/docs/en/plugin-evals.md). It checks that the skill triggers when it should and not when it shouldn't, and that it plans jobs correctly in `--dry-run` mode, so it spends no GPU time:

```bash
claude plugin eval . --allow-tools "Bash(python3 *)" "Bash(ffprobe *)" "Bash(ffmpeg *)"
```

## Models and licenses

talkframe's own code is Apache-2.0. The model weights are downloaded at run time from their publishers on Hugging Face, under their own licenses:
- Wan 2.1 (Wan-AI)
- InfiniteTalk (MeiGen-AI)
- lightx2v
- UMT5, CLIP ViT-H and chinese-wav2vec2-base

Repackaging is by kijai and Comfy-Org. Check those licenses for your use case.
