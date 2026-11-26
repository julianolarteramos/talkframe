---
name: talkframe
description: Turn a still photo of a person plus a voice recording into a realistic talking video. The person speaks in sync with the audio and moves naturally (facial expressions, head, hands, hair, a living background). It runs InfiniteTalk (Wan 2.1) unattended on a rented GPU through Google Colab, an SSH GPU box, or a local NVIDIA GPU. Use this skill whenever the user wants to make a photo talk, animate a portrait with a voiceover, create a talking head, AI presenter, avatar, spokesperson, narrator or news-anchor clip from an image and audio, lip-sync a picture to speech, or add a speaking host to a reel, explainer or course video, even if they don't say "InfiniteTalk" or "lip sync".
---

# talkframe

Make a portrait photo speak an audio track, with lifelike motion rather than just moving lips.

The heavy lifting happens on a GPU machine. `scripts/talkframe.py` plans the job locally, runs it remotely, and stitches the result. Only Python's standard library and ffmpeg are needed on the user's machine. The GPU machine installs ComfyUI, kijai's WanVideoWrapper and ~38 GB of fp8 models by itself (`scripts/remote/job.py`).

## Workflow

1. **Collect the inputs.**
   - **Image:** a photo or render of one person, face clearly visible. Any orientation; the video keeps the image's aspect ratio. Hands and shoulders in frame give the model room to gesture.
   - **Audio:** speech in any language (wav, mp3, m4a…). Clean voice works best, so strip music beds first if possible.
   - **Output:** if the user says where the video will be used (a strip at the bottom of a vertical reel, a 16:9 YouTube insert, a square post), pick the size for it in step 3.

2. **Look at the image and write the scene prompt** in English: who they are, where they are, what they hold, and how they should move while talking. A specific prompt is what makes the motion feel natural, so follow `references/prompting.md`. Don't describe lip shapes; the audio drives the mouth.

3. **Choose the model, size and quality.**
   - `--model infinitetalk` (default): fits Colab's 40 GB A100, the cheapest per clip. Hands often stay at rest.
   - `--model longcat` (LongCat-Video-Avatar 1.6, MIT): steadiest hands and identity of 6 models tested. It uses the official 50/50 distill-LoRA mix, because 100 % DMD exaggerated the mouth (`--lc-mix-alpha` / `--lc-mix-steps` tune it). It needs ~62 GB of VRAM, so it defaults to `--backend runpod` with 80 GB GPUs. Measured: 10 s of video in ~10 min end to end on an A100 80GB, ~US$0.35.
   - `--quality high` is the default: 6 steps, plain attention. It had the best lip-sync of 17 test renders (SyncNet LSE-C 7.98 / LSE-D 6.64).
   - `--quality fast` (4 steps + SageAttention + fp8 matmuls) is ~35 % quicker, but measurably worse on lip-sync (LSE-C 7.02 / LSE-D 7.62), and some syllables degrade. Use it only for drafts.
   - `--steps`, `--attention` and `--fp8-fast` override single parts.
   - Default `--resolution 480`: about 0.4 MP (848×480 for 16:9) and fastest. Good for overlays or anything shown smaller than ~1080 px wide.
   - `--resolution 720`: sharper, about 2–3× slower.
   - `--size WxH` forces an exact frame. If the photo's shape differs, talkframe crops it first. When the video is wider than the photo (a vertical portrait into a landscape strip), it keeps a band centred at 30 % of the photo's height, which usually holds the whole face (`--crop-y` adjusts it). When the video is taller, it keeps the horizontal centre. `PLAN … photo_crop=x,y,w,h` shows the box. If the face matters, look at the photo and adjust `--crop-y` so eyes and mouth are inside.

4. **Check the backend once per session.**
   ```bash
   python3 ${CLAUDE_SKILL_DIR}/scripts/talkframe.py check --backend colab
   ```
   If it reports a problem, read `references/backends.md` for the one-time setup (Colab CLI + ADC login, an SSH host, or a local GPU) and walk the user through it. Login steps need the user in a browser, so ask them to run those commands.

5. **Render.** Call the script directly, one command per call. It creates its own folders and prints what it did, so no `mkdir`, loops or pre-cropping are needed. Rendering takes minutes, so run it in the background and watch the log instead of blocking:
   ```bash
   python3 ${CLAUDE_SKILL_DIR}/scripts/talkframe.py generate \
     --image <photo> --audio <voice> --out <dir>/<name>.mp4 \
     --prompt "<scene prompt>" [--resolution 480|720] [--backend colab|runpod|ssh|local] [--host user@gpu-box]
   ```
   - Exit code **75** means no GPU was free: Colab after all retries, or none of the RunPod GPU types in stock. Nothing is broken; try again later, or rerun the same command with the other backend (`--backend runpod` / `--backend colab`). Auth, balance and other permanent problems fail fast with exit 1 instead.
   - Several clips in a row: pass `--keep` on the first one, then `--session <name>` (printed at the end) on the next ones. That reuses the warm VM and skips the ~5 min of setup. The last call without `--keep` stops the VM. As a safety net, a kept VM is stopped automatically after `--keep-max` idle minutes (default 15). The timer pauses while a `--session` job is using it.
   - **One render per input set.** If `--out` already exists and its `.report.json` has the same input fingerprint (photo, audio, prompt, size, steps, seed…), talkframe returns at once without touching a GPU. Build edits, retries and variants on that video. Pass `--force` only when a new render is truly wanted.
   - Add `--dry-run` to only plan. It spends no GPU time and prints a one-line `PLAN output_size=… segments=…` summary, then the full plan. The plan files (`plan.json`, `in/jobs.json`) go next to the output, or to a temp dir if that place is read-only; the printed `workdir` says where. Use it to show the user the plan, or when they only asked what would happen.
   - Audio longer than 60 s is split at pauses into ≤60 s segments, because InfiniteTalk's colors drift on very long takes. Each segment restarts from the photo, and the original audio is laid back over the joined picture, so the sound has no seams.
   - Each rendered part is padded to its exact audio length before joining, because the model returns a few frames short. That keeps later segments in sync and the last word intact.
   - Picture and voice are kept as rendered (`--video-delay 0`). Measured on real reels, the mouth starts ~1 frame before the voice, which is natural. A 4-frame delay was tried as the default and made the lips visibly late. Use `--video-delay N` only to nudge a specific clip.
   - Timing on a Colab A100:
     - A fresh VM spends ~5 min installing and downloading.
     - Rendering then takes ~20–25 s per second of video at 480p.
     - A 30 s clip is ~15 min end to end and ~2 Colab compute units.

6. **Check the result before reporting.**
   - Make sure the mp4 exists and its duration matches the audio (`ffprobe`).
   - Pull 6–12 frames into a contact sheet (`ffmpeg -i out.mp4 -vf "fps=1,scale=320:-2,tile=4x3" sheet.jpg`) and look at it. Watch for identity drift, broken hands, or a frozen face.
   - Then tell the user:
     - where the file is;
     - duration and size;
     - wall time and compute units, from `<name>.report.json`;
     - anything that looked off.

## Tuning when the result is weak

| Symptom | Try |
|---|---|
| Lips too subtle or out of sync | `--audio-scale 1.3` (up to ~1.5) |
| Too little body motion | A more specific prompt about gestures and the setting; a wider image with hands visible |
| Face changes identity over a long clip | `--max-segment 30` |
| A different take of the same clip | `--seed <n>` |
| Softness or less detail than wanted | `--resolution 720` (more steps did not help: 8 measured worse than 6) |
| Unstable hands or face drifting from the photo | `--model longcat` (steadiest hands and identity in tests; needs an 80 GB GPU, runs on RunPod) |
| Logo or text in the photo wobbles (mic flag, sign, printed name) | `--protect-region X,Y,W,H` (photo pixels; see below) |
| Mouth hangs open during short pauses | Partly handled: silences get a quiet noise floor before the model (measured ~20–30% less opening). The rest is the model anticipating the next word; avoid pure-zero gaps when editing the voice track. `--no-noise-floor` turns the floor off. |

### Keeping a logo exact

The model redraws every pixel, so small text such as a microphone flag's logo drifts from frame to frame. `--protect-region` tracks a flat area through the video and pastes the photo's original pixels back, relit per frame and feathered at the edges (`scripts/protect.py`, OpenCV ECC homography). Frames where tracking is unsure, such as a hand passing in front, are left as generated.

1. Find the box in the photo's own pixel coordinates. Look at the image, or threshold the bright card with OpenCV.
2. Include the whole card or sign, not just the letters. The edges are what make tracking stable.
3. The step needs `numpy` and `opencv-python-headless`. If the Python running talkframe lacks them, it falls back to `uv run --with ...` on its own. Pass `--protect-python` only to force a specific interpreter.
4. Check the result: crop the region from a few frames, side by side.

## When things fail

The script stops with a clear message. The causes seen so far and their fixes (no free Colab GPU, auth, VM reclaimed, dependency breakage) are in `references/troubleshooting.md`. Read it before improvising. The working dependency pins live in `scripts/remote/job.py`.
