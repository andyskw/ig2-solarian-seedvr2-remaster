# Shot-aware SeedVR2 video remaster pipeline

A practical, **scene-by-scene** pipeline for AI-remastering short videos (intros, trailers, old
game/CGI footage) to a clean 1080p with [SeedVR2](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) —
tuned **per shot**, frame-accurate, and **resume-safe**. Runs on a rented CUDA GPU (fast) **or** on a
local AMD iGPU via ROCm (slow but free).

This repo is the distilled result of a long, messy real-world experiment (a 25-year-old game intro).
The full blow-by-blow — including every dead end — is in [`EXPERIMENTS.md`](EXPERIMENTS.md). The point
of this README is so you **don't have to start from zero**.

> ⚠️ This repo contains **code and method only**. The remastered video itself is a derivative of
> copyrighted footage — not included. Bring your own source.

---

## TL;DR — the lessons that actually mattered

1. **Source quality beats the model.** The single biggest jump came from swapping a 360p source for a
   1080p one of the *same* footage (different language dub) and muxing the desired audio back at the end.
   Pick the best available source; treat audio as separable.
2. **Process per shot, not per video.** Split at every cut (frame-accurate), run SeedVR2 with
   `batch_size = shot length` for max temporal coherence, and **tune per shot by content**.
3. **The flicker lever is `latent_noise_scale`, not `batch_size`.** Small, fast-moving objects get
   "re-hallucinated" every frame. Bigger batches plateau; `latent_noise` (low on faces to keep detail,
   higher on fast motion to kill flicker) is what works. Deflicker filters and RIFE do **not** fix it.
4. **Scene detection needs a human.** Content/adaptive detectors miss soft **dissolves** and either
   over-segment fast action (explosions = false cuts) or miss cuts between visually-similar shots.
   Calibrate a detector to a few hand-marked cuts, then **verify the full list with a contact sheet**.
5. **The VAE decode is the bottleneck** (fixed fp16 VAE), so the DiT model size/quantization barely
   changes total runtime. On low-VRAM GPUs use `--vae_decode_tiled`.
6. **Hardware reality:** an AMD 890M iGPU works (~40 s/frame → days) but a rented Blackwell card does
   the whole thing in ~2 hours for a couple of dollars, and better (no VRAM limits). See numbers below.

---

## What the pipeline does

```
source.mp4
  └─(detect_shots.py)──> shot_config.tsv   (cuts + per-shot latent_noise)
       └─(seedvr2_pipeline.py split)──> shots/shot_NN.mp4   (frame-exact, pre-cleaned)
            └─(… process)──> shots_sr/shot_NN.mp4           (per-shot SeedVR2)
                 └─(… concat)──> full_sr.mp4                (silent 1080p master)
                      └─ ffmpeg audio remux ──> final remaster
```

- **Frame-exact split** (`trim=start_frame:end_frame`) — avoids the ~1-frame-per-cut drift you get from
  time-based `-ss/-to`, which otherwise breaks lip-sync over a few minutes.
- **Pre-clean** before SR: light `hqdn3d` + a touch of `unsharp` (don't over-deblock a clean source).
- **Per-shot SeedVR2** with per-shot `latent_noise_scale` and batch sized to the shot.
- **Resume-safe:** finished shots are skipped, so a crash/interruption costs at most one shot.

---

## Requirements

- `ffmpeg` / `ffprobe`
- Python 3.10+ with [PySceneDetect](https://www.scenedetect.com/) (`pip install "scenedetect[opencv]"`)
  for shot detection, plus Pillow for the contact sheet.
- [`ComfyUI-SeedVR2_VideoUpscaler`](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) (v2.5.23+)
  cloned somewhere; its `inference_cli.py` is what we drive.
- A GPU: NVIDIA (CUDA 12.8+ for Blackwell) **or** AMD ROCm.

---

## Path A — rented CUDA GPU (recommended)

A 96 GB Blackwell card (e.g. RTX PRO 6000 on vast.ai, ~$1.1/h) ran the whole 3.5-min intro in **2h 21m
(~$2.7)** — no tiling, whole-shot batches.

```bash
# on a fresh instance (image: a recent PyTorch CUDA 12.8 build; Blackwell = sm_120 needs cu128)
export SEEDVR2_NODE=$PWD/ComfyUI-SeedVR2_VideoUpscaler
bash setup_gpu.sh                       # installs torch(cu128 if needed) + node deps + ffmpeg
python detect_shots.py source.mp4       # -> shot_config.tsv + contact sheet; VERIFY & edit it
python seedvr2_pipeline.py source.mp4 all   # split -> per-shot SR -> concat -> full_sr.mp4
# add your audio:
ffmpeg -i full_sr.mp4 -i audio_source.mp4 -map 0:v -map 1:a -c:v copy -c:a aac -shortest final.mp4
```
Tip: with ≥48 GB VRAM, set `BATCH_CAP=361` to run every shot in a single batch (max coherence).

## Path B — local AMD iGPU (ROCm, gfx1150 / Strix Point)

Works, but ~40 s/frame (days for a few minutes of video). Key environment:

```bash
# self-contained ROCm wheel works even without a system ROCm install:
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ torch torchaudio torchvision
export HSA_OVERRIDE_GFX_VERSION=11.5.1            # 11.0.0 fails with hipErrorInvalidImage
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export MIOPEN_FIND_MODE=2                         # else MIOpen hangs forever on the first VAE conv
export TILED=1                                    # --vae_decode_tiled, required on low/shared VRAM
python seedvr2_pipeline.py source.mp4 all
```

---

## Configuration: `shot_config.tsv`

One row per shot (tab-separated). `detect_shots.py` generates a draft; **you edit it**.

```
shot	start	end	content	latent_noise
1	0.00	11.00	opening	0.15
2	11.00	22.59	ship	0.20
...
```
Suggested `latent_noise` by content: faces **0.05**, characters 0.06–0.08, landscape/interior **0.10**,
fast space/action **0.15**, small fast objects (the worst flicker) **0.20**.

---

## Performance (reference numbers)

| Run (5 s / ~150 frames) | Time | Note |
|---|---|---|
| 3B-FP8, AMD 890M iGPU | 1:16 | ~33 s/frame, VAE-bound |
| 7B-Q4, AMD 890M iGPU | 1:22 | ~same (DiT size barely matters) |
| 3B-FP8, latent_noise=0.15 | 1:42 | flicker control |
| **Full 3.5-min intro, RTX PRO 6000 96 GB** | **2h 21m (~$2.7)** | encode ~28 s/batch vs iGPU ~12 **min**/batch |

---

## Credits & license

- Upscaler: [SeedVR2 / ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)
  (ByteDance Seed · NumZ · AInVFX).
- Shot detection: [PySceneDetect](https://www.scenedetect.com/).
- This pipeline code: MIT (see [`LICENSE`](LICENSE)).
- **Copyright:** the example targets a copyrighted game intro; no video is distributed here. Whatever you
  process is your responsibility.
