#!/usr/bin/env python3
"""Quality gate: ASR-check every local word mp3 and regenerate bad clips.

qwen3-tts is stochastic: a few clips hallucinate (non-English or wrong word).
This sweeps all audio via qwen3-asr-flash (base64, no hosting needed), flags
clips whose transcript is not the target English word, and re-synthesizes just
those (single first, then double as fallback) until clean.

Usage: DASHSCOPE_API_KEY=... python3 tools/verify_fix.py [--voice Serena] [--repeat 1]
"""
import os, re, sys, json, base64, subprocess, tempfile, threading, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO = os.path.join(ROOT, "audio")
EP = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
MODEL = "qwen3-tts-flash"

def load_key():
    k = os.environ.get("DASHSCOPE_API_KEY")
    if k: return k
    for p in (os.path.join(ROOT, "bailian.env"), os.path.join(os.getcwd(), "bailian.env")):
        if os.path.isfile(p):
            for line in open(p):
                if line.startswith("DASHSCOPE_API_KEY="): return line.split("=", 1)[1].strip()
    raise SystemExit("missing DASHSCOPE_API_KEY")

KEY = load_key()
VOICE, REPEAT = "Serena", 1
for i, a in enumerate(sys.argv):
    if a == "--voice": VOICE = sys.argv[i + 1]
    if a == "--repeat": REPEAT = int(sys.argv[i + 1])

def slug(en): return en.lower().replace("'", "").replace(" ", "-")
def norm(s): return re.sub(r"[^a-z0-9]", "", s.lower())

def words():
    src = open(os.path.join(ROOT, "index.html"), encoding="utf-8").read()
    return list(dict.fromkeys(re.findall(r"\[['\"]([A-Za-z][A-Za-z' \-]*)['\"],'", src)))

def post(body):
    for a in range(6):
        try:
            return json.load(urllib.request.urlopen(
                urllib.request.Request(EP, data=json.dumps(body).encode(),
                                       headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}), timeout=90))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and a < 5:
                time.sleep(2 ** a + 1); continue
            raise
        except Exception:
            if a < 5: time.sleep(2 ** a + 1); continue
            raise

def asr_bytes(data):
    throttle()
    b64 = base64.b64encode(data).decode()
    d = post({"model": "qwen3-asr-flash",
              "input": {"messages": [{"role": "user", "content": [{"audio": f"data:audio/mp3;base64,{b64}"}]}]},
              "parameters": {"asr_options": {"enable_lid": True, "enable_itn": False}}})
    m = d["output"]["choices"][0]["message"]
    lang = (m.get("annotations") or [{}])[0].get("language", "?")
    txt = "".join(c.get("text", "") for c in m["content"])
    return lang, txt

def is_clean(lang, txt, en):
    # Any CJK/Japanese/Korean in the transcript means it rambled non-English.
    if any(0x4E00 <= ord(c) <= 0x9FFF or 0x3040 <= ord(c) <= 0x30FF or 0xAC00 <= ord(c) <= 0xD7AF for c in txt): return False
    nw, nt = norm(en), norm(txt)
    if not nt: return False
    if len(nt) > len(nw) + 8: return False          # rambling / extra sentence
    return nw in nt or nt in nw                      # target word present (allow ASR phonetic noise)

_gate = threading.Lock(); _last = [0.0]
def throttle():
    with _gate:
        w = 0.6 - (time.time() - _last[0])
        if w > 0: time.sleep(w)
        _last[0] = time.time()

def synth_bytes(en, repeat):
    text = ". ".join([en] * repeat) + "."
    for a in range(3):
        throttle()
        try:
            d = post({"model": MODEL, "input": {"text": text, "voice": VOICE, "language_type": "English"}})
            return urllib.request.urlopen(d["output"]["audio"]["url"], timeout=90).read()
        except Exception:
            if a < 2: time.sleep(1 + a); continue
            raise

def to_mp3(wav):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t:
        t.write(wav); tmp = t.name
    out = tmp + ".mp3"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmp, "-ac", "1", "-ar", "24000", "-b:a", "48k", out],
                   check=True, capture_output=True)
    data = open(out, "rb").read()
    os.unlink(tmp); os.unlink(out)
    return data

def check(en):
    p = os.path.join(AUDIO, slug(en) + ".mp3")
    try:
        lang, txt = asr_bytes(open(p, "rb").read())
        return en, ("clean" if is_clean(lang, txt, en) else "bad"), lang, txt
    except Exception as e:
        return en, "err", "ERR", str(e)[:60]

def fix(en):
    for repeat in (REPEAT, 2, 3):
        for attempt in range(2):
            try:
                data = to_mp3(synth_bytes(en, repeat))
                lang, txt = asr_bytes(data)
                if is_clean(lang, txt, en):
                    open(os.path.join(AUDIO, slug(en) + ".mp3"), "wb").write(data)
                    return en, True, f"repeat{repeat} [{lang}] {txt[:40]}"
            except Exception:
                pass
    return en, False, "still bad"

def main():
    ws = words()
    print(f"sweep {len(ws)} words voice={VOICE}", flush=True)
    bad = []; err = []; done = 0
    with ThreadPoolExecutor(3) as ex:
        for f in as_completed([ex.submit(check, w) for w in ws]):
            en, st, lang, txt = f.result(); done += 1
            if st == "bad": bad.append(en); print("BAD", en, f"[{lang}] {txt[:50]}", flush=True)
            elif st == "err": err.append(en)
            if done % 100 == 0: print(f"sweep {done}/{len(ws)} bad={len(bad)} err={len(err)}", flush=True)
    # retry transient errors once, sequentially-ish
    if err:
        print(f"retry {len(err)} transient errors", flush=True)
        with ThreadPoolExecutor(2) as ex:
            for f in as_completed([ex.submit(check, w) for w in err]):
                en, st, lang, txt = f.result()
                if st == "bad": bad.append(en); print("BAD", en, f"[{lang}] {txt[:50]}", flush=True)
                elif st == "err": print("SKIP-err", en, txt, flush=True)
    print(f"sweep done bad={len(bad)}: {bad}", flush=True)
    fixed = 0
    with ThreadPoolExecutor(3) as ex:
        for f in as_completed([ex.submit(fix, w) for w in bad]):
            en, ok, msg = f.result()
            print("FIX" if ok else "FAIL", en, msg, flush=True)
            fixed += ok
    print(f"DONE bad={len(bad)} fixed={fixed} remaining={len(bad)-fixed}", flush=True)

if __name__ == "__main__":
    main()