#!/usr/bin/env python3
"""
Shot-aware SeedVR2 remaster pipeline (generic).

Splits a video into shots (frame-exact), runs SeedVR2 per shot with a per-shot
latent_noise_scale, then concatenates a silent 1080p master. Resume-safe.

Usage:
    python seedvr2_pipeline.py SOURCE.mp4 [split|process|concat|all]

Config (TSV, default ./shot_config.tsv) — one row per shot, tab-separated, header required:
    shot    start   end     content   latent_noise
    1       0.00    11.00   opening   0.15
    2       11.00   22.59   ship      0.20
('content' is just a human label and is ignored.)

Environment:
    SEEDVR2_NODE   path to a cloned ComfyUI-SeedVR2_VideoUpscaler (has inference_cli.py)  [required]
    OUTDIR         work/output dir (default: ./out)
    CONFIG         path to shot_config.tsv (default: ./shot_config.tsv)
    MODEL          DiT model (default: seedvr2_ema_3b_fp8_e4m3fn.safetensors)
    RESOLUTION     short-side target (default: 1080)
    BATCH_CAP      max frames per batch, rounded to 4n+1 (default: 161; raise on big-VRAM GPUs)
    TILED          "1" -> add --vae_decode_tiled (needed on low/shared VRAM, e.g. AMD iGPU)
    PRECLEAN       ffmpeg vf for pre-clean (default: light denoise+unsharp; "none" to disable)

For AMD ROCm (gfx1150 etc.) also export before running:
    HSA_OVERRIDE_GFX_VERSION=11.5.1  TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
    MIOPEN_FIND_MODE=2  MIOPEN_USER_DB_PATH=~/.cache/miopen   TILED=1
"""
import os, sys, math, csv, subprocess

SRC      = sys.argv[1] if len(sys.argv) > 1 else None
MODE     = sys.argv[2] if len(sys.argv) > 2 else "all"
NODE     = os.environ.get("SEEDVR2_NODE")
OUTDIR   = os.environ.get("OUTDIR", "./out")
CONFIG   = os.environ.get("CONFIG", "./shot_config.tsv")
MODEL    = os.environ.get("MODEL", "seedvr2_ema_3b_fp8_e4m3fn.safetensors")
RES      = os.environ.get("RESOLUTION", "1080")
BATCH_CAP= int(os.environ.get("BATCH_CAP", "161"))
TILED    = os.environ.get("TILED", "0") == "1"
PRECLEAN = os.environ.get("PRECLEAN", "hqdn3d=1.5:1.5:3:3,unsharp=5:5:0.6:5:5:0.0")
CLIPS, SR = f"{OUTDIR}/shots", f"{OUTDIR}/shots_sr"

def die(msg): print("ERROR:", msg); sys.exit(1)
if not SRC: die("no source video given")
if not NODE: die("set SEEDVR2_NODE to your ComfyUI-SeedVR2_VideoUpscaler path")

def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd[:6]), "...", flush=True)
    subprocess.run(cmd, check=True, **kw)

def probe(path, entries):
    return subprocess.run(["ffprobe","-v","error","-select_streams","v:0",
        "-show_entries",entries,"-of","csv=p=0",path], capture_output=True, text=True).stdout.strip()

def fps_of(path):
    r = probe(path, "stream=r_frame_rate").split("/")
    return int(r[0])/int(r[1]) if len(r)==2 else float(r[0])

def nframes(path):
    return int(probe(path, "stream=nb_read_frames").split("\n")[0] or "0") \
        if False else int(subprocess.run(["ffprobe","-v","error","-select_streams","v:0",
        "-count_frames","-show_entries","stream=nb_read_frames","-of","csv=p=0",path],
        capture_output=True, text=True).stdout.strip())

def load_shots():
    shots = []
    with open(CONFIG) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            shots.append((float(row["start"]), float(row["end"]), float(row["latent_noise"])))
    if not shots: die(f"no shots in {CONFIG}")
    return shots

def frame_bounds(shots, fps, total):
    secs = [shots[0][0]] + [e for (_, e, _) in shots]
    fb = [round(t*fps) for t in secs]
    for i in range(1, len(fb)):
        if fb[i] <= fb[i-1]: fb[i] = fb[i-1]+1
    fb[-1] = total
    return fb

def batch_for(frames):
    if frames <= 5: return 5
    cap = min(frames, BATCH_CAP)
    return max(5, 4*math.ceil((cap-1)/4)+1)

def do_split(shots, fps):
    os.makedirs(CLIPS, exist_ok=True)
    total = nframes(SRC); fb = frame_bounds(shots, fps, total)
    vf = f"trim=start_frame=%d:end_frame=%d,setpts=PTS-STARTPTS" + ("" if PRECLEAN=="none" else f",{PRECLEAN}")
    for i in range(len(shots)):
        clip = f"{CLIPS}/shot_{i+1:03d}.mp4"; A, B = fb[i], fb[i+1]
        if os.path.exists(clip) and nframes(clip) == (B-A): continue
        run(["ffmpeg","-v","error","-i",SRC,"-vf",vf % (A,B),
             "-an","-fps_mode","passthrough","-c:v","libx264","-crf","12","-pix_fmt","yuv420p","-y",clip])
    got = sum(nframes(f"{CLIPS}/shot_{i+1:03d}.mp4") for i in range(len(shots)))
    print(f"SPLIT DONE: {len(shots)} shots, {got} frames (source {total})")

def do_process(shots):
    os.makedirs(SR, exist_ok=True)
    for i, (s, e, ln) in enumerate(shots):
        clip = f"{CLIPS}/shot_{i+1:03d}.mp4"; out = f"{SR}/shot_{i+1:03d}.mp4"
        if os.path.exists(out): print(f"shot {i+1:03d}: exists, skip"); continue
        frames = nframes(clip); b = batch_for(frames); ov = 0 if frames <= BATCH_CAP else 8
        cmd = ["python","inference_cli.py",os.path.abspath(clip),"--output",os.path.abspath(out),
               "--output_format","mp4","--video_backend","ffmpeg","--dit_model",MODEL,
               "--resolution",RES,"--batch_size",str(b),"--uniform_batch_size",
               "--latent_noise_scale",str(ln),"--attention_mode","sdpa"]
        if ov: cmd += ["--temporal_overlap",str(ov)]
        if TILED: cmd += ["--vae_decode_tiled"]
        print(f"=== shot {i+1:03d} ({s:.1f}-{e:.1f}s) frames={frames} batch={b} ln={ln} ===", flush=True)
        run(cmd, cwd=NODE, env=os.environ)
    print("PROCESS DONE")

def do_concat(shots):
    lst = f"{OUTDIR}/concat.txt"
    with open(lst, "w") as f:
        for i in range(len(shots)): f.write(f"file '{os.path.abspath(SR)}/shot_{i+1:03d}.mp4'\n")
    out = f"{OUTDIR}/full_sr.mp4"
    run(["ffmpeg","-v","error","-f","concat","-safe","0","-i",lst,"-c","copy","-y",out])
    print("CONCAT DONE ->", out)

if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    shots = load_shots(); fps = fps_of(SRC)
    if MODE in ("split","all"):   do_split(shots, fps)
    if MODE in ("process","all"): do_process(shots)
    if MODE in ("concat","all"):  do_concat(shots)
