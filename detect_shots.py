#!/usr/bin/env python3
"""
Shot detection + verification contact sheet.

Detects cuts with PySceneDetect (AdaptiveDetector handles motion/action better than a
fixed content threshold), writes a draft shot_config.tsv, and renders a labeled contact
sheet (one thumbnail per shot) so you can VERIFY the list by eye.

Automatic detection is never perfect on real footage:
  - it OVER-segments fast action (explosions read as cuts),
  - it MISSES cuts between visually-similar shots, and
  - it MISSES soft DISSOLVES entirely.
So: calibrate the threshold to a few cuts you know, then edit shot_config.tsv by hand
(merge false cuts, add missed/dissolve cuts) using the contact sheet.

Usage:
    pip install "scenedetect[opencv]" pillow
    python detect_shots.py SOURCE.mp4 [--threshold 2.0] [--min-len 8] [--cols 8]

Outputs: shot_config.tsv  and  shot_contactsheet.png
"""
import argparse, subprocess, os
from scenedetect import detect, AdaptiveDetector

# default latent_noise by simple heuristics — EDIT per shot after reviewing the sheet.
DEFAULT_NOISE = 0.12

def nframes_dur(path):
    out = subprocess.run(["ffprobe","-v","error","-select_streams","v:0",
        "-show_entries","stream=nb_frames","-show_entries","format=duration","-of","default=nw=1",path],
        capture_output=True, text=True).stdout
    dur = [l.split("=")[1] for l in out.splitlines() if l.startswith("duration")]
    return float(dur[0]) if dur else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--threshold", type=float, default=2.0, help="AdaptiveDetector threshold (lower=more cuts)")
    ap.add_argument("--min-len", type=int, default=8, help="min scene length in frames (suppresses flashes)")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--noise", type=float, default=DEFAULT_NOISE)
    ap.add_argument("--config", default="shot_config.tsv")
    ap.add_argument("--sheet", default="shot_contactsheet.png")
    a = ap.parse_args()

    print(f"detecting cuts in {a.source} (adaptive t={a.threshold}, min_len={a.min_len}) ...")
    scenes = detect(a.source, AdaptiveDetector(adaptive_threshold=a.threshold, min_scene_len=a.min_len),
                    show_progress=True)
    end = nframes_dur(a.source) or scenes[-1][1].get_seconds()
    bounds = [0.0] + [s[0].get_seconds() for s in scenes][1:] + [end]
    shots = [(round(bounds[i],2), round(bounds[i+1],2)) for i in range(len(bounds)-1)]
    print(f"{len(shots)} shots detected")

    with open(a.config, "w") as f:
        f.write("shot\tstart\tend\tcontent\tlatent_noise\n")
        for i,(s,e) in enumerate(shots):
            f.write(f"{i+1}\t{s}\t{e}\t\t{a.noise}\n")
    print("wrote", a.config, "-> EDIT IT: merge false cuts, add missed/dissolve cuts, set per-shot latent_noise")

    # contact sheet
    from PIL import Image, ImageDraw, ImageFont
    os.makedirs("/tmp/_shots", exist_ok=True)
    TW, TH = 300, 169
    for i,(s,e) in enumerate(shots):
        subprocess.run(["ffmpeg","-v","error","-ss",f"{(s+e)/2:.2f}","-i",a.source,"-frames:v","1",
            "-vf",f"scale={TW}:{TH}","-y",f"/tmp/_shots/s{i:03d}.png"], check=True)
    cols = a.cols; rows = (len(shots)+cols-1)//cols
    LBL, pad = 20, 2; cw, ch = TW+pad*2, TH+LBL+pad*2
    sheet = Image.new("RGB", (cols*cw, rows*ch), (20,20,20)); d = ImageDraw.Draw(sheet)
    try: font = ImageFont.truetype("DejaVuSans-Bold.ttf", 13)
    except Exception:
        try: font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 13)
        except Exception: font = ImageFont.load_default()
    for i,(s,e) in enumerate(shots):
        r,c = divmod(i, cols); x,y = c*cw+pad, r*ch+pad
        sheet.paste(Image.open(f"/tmp/_shots/s{i:03d}.png"), (x, y+LBL))
        d.rectangle([x,y,x+TW,y+LBL], fill=(0,0,0))
        d.text((x+2,y+3), f"#{i+1} {s:.1f}-{e:.1f}", fill=(255,230,0), font=font)
    sheet.save(a.sheet)
    print("wrote", a.sheet, f"({sheet.size[0]}x{sheet.size[1]}) — review and fix shot_config.tsv")

if __name__ == "__main__":
    main()
