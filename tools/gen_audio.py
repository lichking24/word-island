#!/usr/bin/env python3
"""Batch-generate native-English word pronunciation mp3s via Bailian qwen3-tts-flash.

Key is read from $DASHSCOPE_API_KEY or ./bailian.env (never committed).
Audio is written to ../audio/<slug>.mp3 and committed; the public site plays
these static files so no API key is ever exposed in the browser.

Usage:  DASHSCOPE_API_KEY=sk-... python3 tools/gen_audio.py [--voice Jennifer] [--workers 4]
"""
import os, re, json, sys, time, subprocess, tempfile, threading, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO = os.path.join(ROOT, "audio")
MODEL = "qwen3-tts-flash"
ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

def load_key():
    k = os.environ.get("DASHSCOPE_API_KEY")
    if k: return k
    for p in (os.path.join(ROOT, "bailian.env"), os.path.join(os.getcwd(), "bailian.env")):
        if os.path.isfile(p):
            for line in open(p):
                if line.startswith("DASHSCOPE_API_KEY="):
                    return line.split("=", 1)[1].strip()
    sys.exit("missing DASHSCOPE_API_KEY (env or bailian.env)")

def slug(en):
    return en.lower().replace("'", "").replace(" ", "-")

def words():
    src = open(os.path.join(ROOT, "index.html"), encoding="utf-8").read()
    seen, out = set(), []
    for en in re.findall(r"\[['\"]([A-Za-z][A-Za-z' \-]*)['\"],'", src):
        if en not in seen:
            seen.add(en); out.append(en)
    return out

KEY = load_key()
VOICE = "Jennifer"
WORKERS = 4
FORCE = False
for i, a in enumerate(sys.argv):
    if a == "--voice": VOICE = sys.argv[i + 1]
    if a == "--workers": WORKERS = int(sys.argv[i + 1])
    if a == "--force": FORCE = True

_gate = threading.Lock()
_last = [0.0]
MIN_GAP = 0.35  # ~<=170 req/min, under the 180 RPM limit

def throttle():
    with _gate:
        wait = MIN_GAP - (time.time() - _last[0])
        if wait > 0: time.sleep(wait)
        _last[0] = time.time()

def synth(en):
    body = {"model": MODEL, "input": {"text": en, "voice": VOICE, "language_type": "English"}}
    for attempt in range(4):
        throttle()
        req = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(),
                                     headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
        try:
            d = json.load(urllib.request.urlopen(req, timeout=90))
            url = d["output"]["audio"]["url"]
            wav = urllib.request.urlopen(url, timeout=90).read()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t:
                t.write(wav); tmp = t.name
            out = os.path.join(AUDIO, slug(en) + ".mp3")
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmp, "-ac", "1", "-ar", "24000", "-b:a", "48k", out],
                           check=True, capture_output=True)
            os.unlink(tmp)
            return en, os.path.getsize(out), None
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:200]
            if e.code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(2 ** attempt + 1); continue
            return en, 0, f"HTTP {e.code} {msg}"
        except Exception as e:
            if attempt < 3:
                time.sleep(2 ** attempt + 1); continue
            return en, 0, str(e)[:200]

def main():
    os.makedirs(AUDIO, exist_ok=True)
    ws = words()
    todo = ws if FORCE else [w for w in ws if not os.path.isfile(os.path.join(AUDIO, slug(w) + ".mp3"))]
    print(f"total unique words={len(ws)} already={len(ws)-len(todo)} to_generate={len(todo)} voice={VOICE}", flush=True)
    done = 0; failed = []; total_bytes = 0; t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(synth, w): w for w in todo}
        for f in as_completed(futs):
            en, size, err = f.result()
            done += 1
            if err: failed.append((en, err))
            else: total_bytes += size
            if done % 25 == 0 or done == len(todo):
                el = time.time() - t0
                rate = done / el if el else 0
                eta = (len(todo) - done) / rate if rate else 0
                print(f"progress {done}/{len(todo)} failed={len(failed)} bytes={total_bytes} elapsed={el:.0f}s eta={eta:.0f}s", flush=True)
    print(f"DONE generated={len(todo)-len(failed)} failed={len(failed)} total_mb={total_bytes/1e6:.2f}", flush=True)
    for en, err in failed: print("FAIL", en, err, flush=True)

if __name__ == "__main__":
    main()