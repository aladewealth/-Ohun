#!/usr/bin/env python3
"""Download Google's crowdsourced Yoruba speech (OpenSLR 86, CC BY-SA 4.0) and write a Piper training CSV.

Output: <out>/wavs/*.wav (original 48 kHz; Piper resamples) and <out>/metadata.csv as
    file.wav|speaker|phonemes
where phonemes come from our rule-based Yoruba G2P (voice/yoruba_g2p.py).
"""
import argparse
import csv
import os
import sys
import urllib.request
import zipfile
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from voice.yoruba_g2p import to_ipa  # noqa: E402

BASE = "https://openslr.trmal.net/resources/86/"
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/yoruba")
ap.add_argument("--genders", default="female,male")
ap.add_argument("--min-utts", type=int, default=50, help="drop speakers with fewer clips")
args = ap.parse_args()

os.makedirs(os.path.join(args.out, "wavs"), exist_ok=True)
rows = []
for g in args.genders.split(","):
    tsv = os.path.join(args.out, f"line_index_{g}.tsv")
    if not os.path.exists(tsv):
        urllib.request.urlretrieve(BASE + f"line_index_{g}.tsv", tsv)
    z = os.path.join(args.out, f"yo_ng_{g}.zip")
    if not os.path.exists(z):
        print("downloading", z)
        urllib.request.urlretrieve(BASE + f"yo_ng_{g}.zip", z)
    with zipfile.ZipFile(z) as zf:
        for name in zf.namelist():
            if name.endswith(".wav"):
                dst = os.path.join(args.out, "wavs", os.path.basename(name))
                if not os.path.exists(dst):
                    with zf.open(name) as src, open(dst, "wb") as f:
                        f.write(src.read())
    for line in open(tsv, encoding="utf-8"):
        fid, text = line.rstrip("\n").split("\t", 1)
        spk = f"{g[0]}{fid.split('_')[1]}"
        wav = fid + ".wav"
        if os.path.exists(os.path.join(args.out, "wavs", wav)):
            rows.append((wav, spk, to_ipa(text)))

counts = Counter(r[1] for r in rows)
rows = [r for r in rows if counts[r[1]] >= args.min_utts]
with open(os.path.join(args.out, "metadata.csv"), "w", encoding="utf-8", newline="") as f:
    csv.writer(f, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\").writerows(rows)
spk = sorted({r[1] for r in rows})
print(f"{len(rows)} utterances, {len(spk)} speakers -> {args.out}/metadata.csv")
print("NUM_SPEAKERS", len(spk))
