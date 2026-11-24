# Backends: where the GPU comes from

All three backends run the same `scripts/remote/job.py`. The GPU needs ≥24 GB of VRAM; the fp8 model peaks at ~24 GB at 480p. The job installs ComfyUI and the models on first use and skips those steps when they are already there.

## colab (default): Google Colab via the official CLI

This is the cheapest way to get an A100 with no infrastructure. It needs a Colab plan with compute units (Colab Pro or pay-as-you-go). An A100 burns ~8 compute units per hour; 100 units cost about US$10.

One-time setup, on Linux, macOS or WSL:

```bash
uv tool install google-colab-cli          # or: pip install google-colab-cli
gcloud auth application-default login --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
colab --auth adc whoami                    # shows the account
colab --auth adc usage                     # shows the compute-unit balance
```

- **Login:** use the `gcloud` ADC login rather than the CLI's own oauth2 flow. The oauth2 token lives in the OS keyring, and on headless boxes or WSL the keyring is usually locked, which means a prompt on every call. ADC refreshes itself. `talkframe.py` always passes `--auth adc`.
- **Account:** the Google account must be the one holding the Colab subscription.

What talkframe does on each run:
1. `colab new --gpu A100`, retrying every minute (`--gpu-attempts`) while none is free.
2. Uploads inputs and `job.py`.
3. Runs it with `colab exec --timeout 0`.
4. Downloads the parts and stops the VM, even on failure.
5. Compares the compute-unit balance before and after.

`--keep` leaves the VM up so the next run skips setup. Stop it later with `colab --auth adc stop -s <name>`, because an idle VM still burns units.

Colab specifics:
- **Keep the kernel busy.** Work started in the background over `colab ssh` doesn't count as activity, so Colab reclaims the VM in ~20–30 min and the outputs are lost. That's why the job runs inside the kernel through `colab exec`.
- **Use `--timeout 0`.** `colab exec` gives up after 30 s with no output unless you pass it. `job.py` also prints a line at least every ~25 s.
- **GPU sizes:** the standard A100 VM has 40 GB VRAM, 83 GB RAM and ~235 GB disk. That's enough for 480p and 720p. `--gpu H100` is faster and costs more units per hour. `--gpu L4` (24 GB) works at 480p but is slower.
- **Downloads:** Hugging Face downloads inside Colab run at roughly 300 MB/s, so the ~38 GB of models take a few minutes. Caching them in Drive isn't worth it, and mounting Drive needs an interactive consent anyway.

## ssh: any Linux box with an NVIDIA GPU

Use this for a workstation, a cloud VM (RunPod, Lambda, Vast, Azure or GCP spot) or a lab server you can reach with key-based SSH.

Requirements on the host:
- `git`
- Python 3.10+ with a CUDA build of PyTorch. Point to it with `--remote-python /path/to/venv/bin/python`.
- ~60 GB of free disk.

```bash
python3 talkframe.py check --backend ssh --host user@gpu-box --remote-python ~/venv/bin/python
python3 talkframe.py generate --backend ssh --host user@gpu-box --remote-python ~/venv/bin/python ...
```

Files live in `~/talkframe-work` on the host (`--remote-dir`). The models stay there between runs, so only the first run pays for the download.

## local: this machine has the GPU

```bash
python3 talkframe.py generate --backend local --remote-python python3 ...
```

The work directory is next to the output (`<out>.talkframe/`), including ComfyUI and the models. Pass `--workdir` to keep one shared location across runs.
