---
title: "I AI-remastered a 25-year-old game intro to real 1080p — and learned that the source matters more than the model"
published: true
tags: ai, machinelearning, video, opensource
cover_image: https://raw.githubusercontent.com/andyskw/ig2-solarian-seedvr2-remaster/main/docs/split50_ship_1m29.png
---

I spent way too long remastering the intro of *Imperium Galactica 2 – Solarian*, a space-strategy game from 2000, to a clean 1080p using AI. Along the way I learned a pile of things the hard way — about SeedVR2, about temporal flicker, about running diffusion on a tiny AMD iGPU, and about how little the "big model" actually matters. This is the whole journey, dead ends included, so you don't have to repeat my mistakes.

Everything (code + the generic pipeline) is here: **https://github.com/andyskw/ig2-solarian-seedvr2-remaster**

▶ **Watch the full remaster:** [English](https://youtu.be/zn15PEU9nGY) · [Hungarian dub](https://www.youtube.com/watch?v=gDFm5QmpbvQ)

https://youtu.be/zn15PEU9nGY

![Original vs remaster, ship detail](https://raw.githubusercontent.com/andyskw/ig2-solarian-seedvr2-remaster/main/docs/before_after_crop.png)

## Lesson 0: the source matters more than the upscaler

I had two copies of the same intro: a 360p one (the language dub I wanted) and a 1080p one (a different dub). The 1080p "looked the same" at a glance — but it isn't. A quick sharpness measurement (Laplacian variance) plus zoomed crops showed it carries genuinely more detail (8.7× the bitrate).

So the single biggest quality jump came from **throwing away my preferred 360p source, upscaling the better 1080p one, and muxing the audio I wanted back at the very end.** Picture and audio are separable.
If you take one thing from this post: **feed your upscaler the best source you can find.**

## Getting AI to run on a tiny AMD iGPU

My always-on box has an AMD **Radeon 890M iGPU (gfx1150)** — no ROCm userspace installed, just the kernel driver. It still works, with two non-obvious tricks:

```bash
# self-contained ROCm wheel — no system ROCm needed
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ torch torchaudio torchvision
export HSA_OVERRIDE_GFX_VERSION=11.5.1   # 11.0.0 -> hipErrorInvalidImage; unset -> GPU invisible
export MIOPEN_FIND_MODE=2                 # else MIOpen hangs FOREVER on the first VAE conv
```

That MIOpen hang cost me ~27 minutes of staring at a GPU pinned at 100% with zero progress before I figured out the FAST find-mode. If you run SeedVR2 (or anything MIOpen-heavy) on a Strix-class iGPU, remember those two lines. :D

## SeedVR2, and why the model size barely matters

[SeedVR2](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) gave the most natural result (real surface texture, temporal awareness). I compared 3B vs 7B and... they took almost the same time. Why?

**The fp16 VAE decode is the bottleneck**, not the diffusion transformer. Quantizing or shrinking the DiT changes runtime by a rounding error. The 3B-FP8 looked to me more natural, so that's what I used.

## The flicker hunt: it's `latent_noise`, not `batch_size`

The intro contains a lot of space fight scenes.

Small, fast-moving objects (a little fighter, drifting shadows) flickered — the model re-invents their detail every frame (however, after carefully watching, the original scenes also had a similar flickering -> upscaling only made them more visible). The intuitive fix is a bigger temporal batch... and it helps, then **plateaus**.

The thing that actually worked was **`latent_noise_scale`**: low on faces (to keep detail), higher on fast motion (to suppress the per-frame re-hallucination). Things that did **not** work: a temporal
deflicker filter (not motion-compensated) and RIFE frame interpolation (smooth, but the flicker is baked into the generated frames — and 60 fps made me slightly motion-sick).

Here's the worst offender — the little fighter in the opening space battle (**~0:58** in the video).
Left = original, right = remaster:

![Flickering ship: original vs remaster](https://raw.githubusercontent.com/andyskw/ig2-solarian-seedvr2-remaster/main/docs/flicker_ship.gif)

Oh, and on low/shared-VRAM GPUs a big batch will OOM the VAE decode — `--vae_decode_tiled` fixes that.

## Scene-by-scene, with a human in the loop

For the best result you process **per shot**: split at every cut, run one batch per shot, and tune
`latent_noise` per shot by content (faces ~0.05, fast action ~0.15, tiny fast objects ~0.20). But
getting the cut list was the real fight:

- ffmpeg's scene filter missed cuts between visually-similar space shots and over-segmented explosions.
  A "72-second shot" was actually ~5 shots.
- PySceneDetect's AdaptiveDetector, calibrated against a few hand-marked cuts, did much better.
- The opening "cut" was a **1-second cross-dissolve** — invisible to content detectors; only my eye caught it.

The workflow that worked: auto-detect → render a **contact sheet** (one thumbnail per shot) → verify and hand-edit. Treat automatic scene detection as a "draft".

![Shot contact sheet for verification](https://raw.githubusercontent.com/andyskw/ig2-solarian-seedvr2-remaster/main/docs/shot_contactsheet.png)

The bulk of the hand-editing was *merging* false cuts. The rule I settled on: it's the same shot if the camera/subject continues across the "cut" (an explosion flash, fast motion, a fighter crossing frame);
it's only a new shot on a real change of framing — or a dissolve, which detectors miss entirely.

![Which shots I merged by hand, and why](https://raw.githubusercontent.com/andyskw/ig2-solarian-seedvr2-remaster/main/docs/manual_merge_explainer.png)

One more trap: **split frame-exactly**. Time-based `-ss/-to` added ~1 frame per cut, which drifted ~2 seconds over the video and broke lip-sync. `trim=start_frame:end_frame` gives the exact source frame count. This can save tons of time, and if you catch it only after the whole video is rendered, you will be really pissed off. :) 

![Original vs remaster, face](https://raw.githubusercontent.com/andyskw/ig2-solarian-seedvr2-remaster/main/docs/before_after_face.png)

## By the numbers: a 5-second proving ground

Here's the thing — I didn't discover any of the above on the full 3.5-minute intro. **Every decision was made on a single 5-second clip**, looped through the pipeline over and over on the iGPU. Source choice, 3B vs 7B, fp8 vs Q4 vs fp16, batch 5 → 25 → 49 → 73, tiling on/off, `latent_noise` 0 → 0.1 → 0.15, plus the deflicker and RIFE dead ends — roughly **a dozen full SeedVR2 runs** on that one clip (two of which OOM'd and had to be killed), adding up to **~16 hours of iGPU compute just to dial in the recipe**.

Out of all that came **~15 side-by-side comparison reels** — and the genuinely fun part was watching the same five seconds run 2, 3, even 4 ways at once:

![A few of the comparison reels](https://raw.githubusercontent.com/andyskw/ig2-solarian-seedvr2-remaster/main/docs/comparison_reels.png)

Iterating on a short clip first is the whole trick: cheap enough to be patient, long enough to judge temporal behavior. Only once those 5 seconds looked right did I commit a single minute to the full render.

## Hardware reality: iGPU days vs cloud hours

On the 890M iGPU the full run extrapolated to **~74 hours** (~40 s/frame). So I rented an **RTX PRO 6000 (Blackwell, 96 GB)** on vast.ai for ~$1.1/hour:

- Blackwell (sm_120) needs **CUDA 12.8** torch (PyTorch 2.12 / cuDNN 9.2).
- Encode dropped from ~12 **minutes**/batch to ~**28 seconds**/batch. With 96 GB: no tiling, whole-shot
  batches.
- **The whole 3.5-minute intro: 2h 21m, about $2.70.**

A few cloud gotchas that ate my time: `pkill -f inference_cli.py` matched its own command line and killed my shell; moving "just the home folder" to a new box left the Python deps behind; and the rented box's stale CA bundle made `curl` reject valid certs (`certificate has expired`).

Here's what that full render buys you on a wide shot — the same 50/50 split, remaster on the left:

![Remaster vs original, landscape (50/50 split)](https://raw.githubusercontent.com/andyskw/ig2-solarian-seedvr2-remaster/main/docs/split50_1m57.png)

## Finishing + a YouTube tip

Mux the audio onto the silent master (`-c:v copy`); the frame-exact split keeps it in sync. And when you upload, **upscale to 4K first.** YouTube assigns bitrate by resolution tier, so a 4K upload preserves your 1080p content far better through its re-encode than a native 1080p upload.

## The recipe (use the repo)

1. Pick the best-quality source; keep audio separate.
2. Detect shots → **verify by hand** (contact sheet); watch for dissolves.
3. Frame-exact split → per-shot SeedVR2 (3B-FP8) with **per-shot `latent_noise`**, batch = shot length.
4. Low-VRAM? `--vae_decode_tiled`. Big cloud GPU? whole-shot batches.
5. Concat → mux audio → upload at 4K.

The generic, resume-safe pipeline (CUDA **and** AMD ROCm), the shot detector, and the full writeup are all here:

👉 **https://github.com/andyskw/ig2-solarian-seedvr2-remaster**

Credits: [SeedVR2](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) (ByteDance Seed · NumZ · AInVFX), [PySceneDetect](https://www.scenedetect.com/).