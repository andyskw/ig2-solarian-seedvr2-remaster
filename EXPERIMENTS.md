# How I AI-remastered a 25-year-old game intro to real 1080p — the full journey

A blow-by-blow log of remastering the *Imperium Galactica 2 – Solarian* intro to a clean 1080p with
[SeedVR2](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) — dead ends included. The point is so
you don't have to start from zero. The reusable code is in this repo (see `README.md`).

## Act 0 — The problem, and the most important decision
Two sources existed: a Hungarian-dubbed **360p** (576×360, 25 fps, 0.48 Mbps — heavy macroblocking) and
an English **1080p** (1920×1080, 30 fps, 4.1 Mbps). The 1080p looked "the same" at a glance, but a
sharpness measurement (Laplacian variance) + zoomed crops proved it carries **genuinely more detail**
(8.7× the bitrate). **Lesson #1: source quality beats the upscaler.** We later upscaled the *English*
1080p and muxed the *Hungarian* audio back at the end — audio is separable from picture.

## Act 1 — Getting AI to run on a weird AMD iGPU
Target machine: an AMD **Radeon 890M iGPU (gfx1150, RDNA 3.5)** — no ROCm userspace installed, only the
KFD kernel driver. What worked:
- A self-contained **gfx1151** PyTorch-ROCm wheel (`torch 2.10+rocm7.13`) — no system ROCm needed.
- The magic env var: `HSA_OVERRIDE_GFX_VERSION=11.5.1` (GPU visible, matmul + SDPA run). `11.0.0` fails
  with `hipErrorInvalidImage`; without it the GPU is invisible.
- **MIOpen hung forever** on the VAE's first conv (GPU pinned at 100%, zero progress for ~27 min).
  Fix: `MIOPEN_FIND_MODE=2` (FAST). Great debugging-drama beat.

## Act 2 — Model bake-off, and the bottleneck
- Real-ESRGAN (ncnn-vulkan) baseline: `animevideov3` cleanest, `x4plus` sharper but noisier; both ≫ bicubic.
- **SeedVR2 3B-FP8** won — real surface texture, temporal awareness. ~33 s/frame on the iGPU.
- 3B vs 7B (Q4): nearly identical *runtime* — because **the fp16 VAE decode is the bottleneck**, not the
  DiT. Model size/quantization barely move total time. 3B-FP8 looked the most natural.

## Act 3 — The big swap (360p → 1080p source)
The Hungarian 360p and English 1080p are **frame-aligned** (verified at several timestamps). So we ran
SeedVR2 on the **English 1080p** (pre-cleaned with a light `hqdn3d` + gentle `unsharp` — don't
over-deblock a clean source). Result vs the 360p-sourced version: dramatically better. The only thing no
upscaler can fix: **motion blur baked into the original** fast frames.

## Act 4 — The flicker hunt: batch_size vs latent_noise
Small, fast-moving objects (a little ship, shadows) flickered — the diffusion model re-invents their
detail every frame.
- `batch_size=5` → strong flicker. `batch_size=73` (first try) → **OOM / swap thrash** (the big batch
  blew up VAE-decode memory). Fix: `--vae_decode_tiled`.
- `batch=25 → 49` with tiling: visibly better, then **plateaus**. Bigger batch is *not* the cure.
- **The real lever is `latent_noise_scale`** (README of the node frames it as artifact reduction): low on
  faces (keep detail), higher on fast motion (kill flicker). `0.1`–`0.15` made a big difference.
- Dead ends: `atadenoise` (not motion-compensated → no help), and **RIFE 60 fps** (smooth, but doesn't
  fix the flicker; also a bit nauseating).

## Act 5 — Scene-by-scene, with a human in the loop
For best coherence: split at every cut (frame-accurate), one batch per shot, and **tune latent_noise per
shot by content** (faces 0.05, characters 0.06–0.08, landscape/interior 0.10, fast action 0.15, tiny
fast objects 0.20). Getting the cut list was the hard part:
- ffmpeg's scene filter at a fixed threshold missed cuts between similar space shots (and over-segmented
  explosions). A **72-second "shot" turned out to be ~5 shots.**
- **PySceneDetect AdaptiveDetector**, calibrated against a handful of hand-marked cuts, matched well.
- The opening "cut" at ~11 s was actually a **~1-second cross-dissolve** — invisible to content
  detectors; only the human eye caught it.
- Workflow that worked: auto-detect → render a **contact sheet (one thumbnail per shot)** → verify and
  hand-edit (merge false cuts, add missed/dissolve cuts). Final: **58 shots.**
- **Frame-exact split matters:** time-based `-ss/-to` added ~1 frame per cut (≈2 s drift over the video →
  broken lip-sync). `trim=start_frame:end_frame` gives exactly the source frame count.

## Act 6 — Hardware reality: iGPU days vs cloud hours
On the 890M iGPU the full run extrapolated to **~74 hours** (~40 s/frame). So we rented an **RTX PRO 6000
(Blackwell, 96 GB)** on vast.ai (~$1.1/h):
- Blackwell (sm_120) needs **CUDA 12.8 torch** (PyTorch 2.12 / cuDNN 9.2).
- Encode ~**28 s/batch** vs the iGPU's ~**12 minutes/batch**. With 96 GB, no tiling, whole-shot batches.
- **Whole 3.5-min intro: 2h 21m, ~$2.7.** Same code, resume-safe, so an interruption costs one shot.

Cloud gotchas worth knowing:
- `pkill -f inference_cli.py` **matched its own command line** and killed the calling shell (SSH dropped
  with 255) — kill by explicit **PID** instead.
- Moving "just the home folder" to a new box left the Python deps behind (`ModuleNotFoundError: cv2`) —
  re-run setup.
- Upload from the rented box failed with `curl: (60) certificate has expired` (stale CA bundle) →
  `curl -k` or `update-ca-certificates`.
- The **SSH port ≠ the Jupyter port** (look for the container's `… -> 22/tcp` mapping).

## Act 7 — Finishing
- **Audio:** muxed the Hungarian track onto the silent master (`-c:v copy`); frame-exact split kept it in
  sync.
- **YouTube tip:** upload **4K** (upscale the 1080p master) so YouTube assigns the 4K bitrate tier — your
  1080p survives its re-encode far cleaner than a native 1080p upload.

## Numbers
| Run (5 s clip unless noted) | Time | Note |
|---|---|---|
| 3B-FP8, AMD 890M iGPU | 1:16 | ~33 s/frame, VAE-bound |
| 7B-Q4, AMD 890M iGPU | 1:22 | ~same (DiT size barely matters) |
| 3B-FP8 + latent_noise=0.15 | 1:42 | flicker control |
| **Full intro, RTX PRO 6000 96 GB** | **2h 21m (~$2.7)** | encode ~28 s/batch vs iGPU ~12 min/batch |

## The recipe (TL;DR)
1. Pick the best-quality source; keep audio separate.
2. Detect shots → **verify by hand** (contact sheet); watch for dissolves.
3. Frame-exact split → per-shot SeedVR2 (3B-FP8) with **per-shot latent_noise**, batch = shot length.
4. Low-VRAM? `--vae_decode_tiled`. Big-VRAM cloud? whole-shot batches.
5. Concat → mux audio → upload at 4K.

Credits: SeedVR2 (ByteDance Seed · NumZ · AInVFX), PySceneDetect.
