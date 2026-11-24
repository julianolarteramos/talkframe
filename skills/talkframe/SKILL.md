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

3. **Choose the size.**
   - Default `--resolution 480`: about 0.4 MP (848×480 for 16:9) and fastest. Good for overlays or anything shown smaller than ~1080 px wide.
   - `--resolution 720`: sharper, about 2–3× slower.
   - `--size WxH` forces an exact frame. The image is center-cropped to that aspect.

4. **Check the backend once per session.**
   ```bash
   python3 ${CLAUDE_SKILL_DIR}/scripts/talkframe.py check --backend colab
   ```
   If it reports a problem, read `references/backends.md` for the one-time setup (Colab CLI + ADC login, an SSH host, or a local GPU) and walk the user through it. Login steps need the user in a browser, so ask them to run those commands.

5. **Render.** This takes minutes, so run it in the background and watch the log instead of blocking:
   ```bash
   python3 ${CLAUDE_SKILL_DIR}/scripts/talkframe.py generate \
     --image <photo> --audio <voice> --out <dir>/<name>.mp4 \
     --prompt "<scene prompt>" [--resolution 480|720] [--backend colab|ssh|local] [--host user@gpu-box]
   ```
   - Add `--dry-run` to only plan. It writes `plan.json` and `in/jobs.json` next to the output and spends no GPU time. Use it to show the user the plan, or when they only asked what would happen.
   - Audio longer than 60 s is split at pauses into ≤60 s segments, because InfiniteTalk's colors drift on very long takes. Each segment restarts from the photo, and the original audio is laid back over the joined picture, so the sound has no seams.
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
| Softness | `--resolution 720`, or `--steps 8` |

## When things fail

The script stops with a clear message. The causes seen so far and their fixes (no free Colab GPU, auth, VM reclaimed, dependency breakage) are in `references/troubleshooting.md`. Read it before improvising. The working dependency pins live in `scripts/remote/job.py`.
