# Troubleshooting

These are real failures hit while building talkframe (November 2026), with the fix for each.

| Message / symptom | Cause | Fix |
|---|---|---|
| `could not get a Colab A100` / `Service Unavailable` on `colab new` | No free A100 in the pool | Wait and retry (`--gpu-attempts 30`), or use `--gpu H100` or `--gpu L4` |
| `colab auth failed` / HTTP 401 | ADC missing the Colab scopes, or a different Google account | Re-run the `gcloud auth application-default login --scopes=...` line in `backends.md` |
| `colab auth failed` the day after it worked (`No valid default credentials`) | Google Workspace accounts can force re-authentication every N hours (admin "session control"), which kills the ADC refresh token | Re-run the ADC login. For unattended use, ask the Workspace admin to exempt the account or use a personal Google account for Colab |
| Keyring "is locked" prompts | CLI's oauth2 mode on a headless box or WSL | Use ADC; the script already passes `--auth adc` |
| `Timeout waiting for output: no message from the kernel for 30s` | `colab exec` without `--timeout 0` | Already handled by the script; add it if you call `colab exec` yourself |
| VM disappears mid-render; `session not found` | Colab reclaimed an idle kernel (work was started outside the kernel) | Run the job through `colab exec`, as the script does |
| `'NoneType' object is not subscriptable` at `hidden_states[1:]` | transformers 5.x changed wav2vec2 outputs | Pin `transformers==4.57.6` (job.py does) |
| `cannot import name 'resolve_revision' from 'huggingface_hub'` → WanVideoWrapper `IMPORT FAILED` | diffusers too new for the hub version that transformers 4.x pulls in | Pin `diffusers==0.35.2` |
| `No module named 'torchsde'` when ComfyUI starts | The torch lines were filtered out of ComfyUI's requirements | job.py installs `torchsde` explicitly |
| Process killed (exit 137) while loading Wan | Official InfiniteTalk repo loads fp32 weights (~65 GB) into RAM | Use the ComfyUI fp8 path (what talkframe does). The official repo's int8 quanto path fits but runs ~125 s per step on an A100, about 14× slower. |
| `pkill -f <pattern>` kills your own SSH command | The pattern also matches the `bash -c` running it | Anchor the pattern (`^python3 main.py`) or kill by PID |
| Lips barely move | Quiet or music-heavy audio | Clean or normalize the voice; `--audio-scale 1.3` |

To debug on Colab, rerun with `--keep`, then:

```bash
colab --auth adc ssh -s <session> -- 'tail -50 /content/talkframe/comfy.log'
```

Remember to `colab stop` afterwards.

When a dependency breaks again:

1. Run `python main.py --quick-test-for-ci` inside the `ComfyUI` folder.
2. Read the `IMPORT FAILED` lines in its output.
3. Adjust `PINS` in `scripts/remote/job.py` until WanVideoWrapper imports cleanly.
