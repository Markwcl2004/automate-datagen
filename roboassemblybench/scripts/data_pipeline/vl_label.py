"""Instruction labelling: a vision-language model describes the plug, the socket and how they mate, from the 4 close-ups
of label_crops.py, 3 times per task with the images in different orders (temperature 0); take the majority, map the shapes
to a controlled vocabulary and review by hand (see examples/instruction_labels_*.json).
env: LLMAPI_BASE_URL (OpenAI-compatible, ending in /v1), LLMAPI_KEY, [LLMAPI_MODEL=qwen3-vl-235b LLMAPI_WORKERS=2]
usage: vl_label.py <crops dir> <out jsonl> <task>..."""
import base64, concurrent.futures as cf, json, os, random, re, sys, time, urllib.error, urllib.request
URL = os.environ["LLMAPI_BASE_URL"].rstrip("/") + "/chat/completions"; KEY = os.environ["LLMAPI_KEY"]; MODEL = os.environ.get("LLMAPI_MODEL", "qwen3-vl-235b")
D, OUT = sys.argv[1], sys.argv[2]
VIEWS = {"front_parts": "front camera close-ups side by side: LEFT the PLUG (white part) standing on the table before it is picked up, RIGHT the SOCKET (green part) fixed on the table",
         "front_mating": "front camera, side view half-way through the assembly: the plug held by the gripper meeting the socket",
         "wrist_grasp": "wrist camera looking straight down at the plug just before the gripper closes on it",
         "wrist_insert": "wrist camera looking straight down at the moment the held plug starts to go into / onto the socket"}
HEAD = "These are close-up images from one robot assembly episode in simulation. Colours are rendering colours only; judge geometry."
TAIL = """Decide how the two parts mate:
- "plug_into_socket": a solid plug goes INTO a hole or opening of the socket;
- "plug_onto_socket": a hollow plug (ring, sleeve, tube) slides OVER / ONTO a post or shaft of the socket.
Answer with JSON only:
{"plug_shape": "<short noun phrase>", "plug_details": "<cross-section, hollow or solid, holes, flanges, steps, teeth>",
 "socket_shape": "<short noun phrase>", "socket_details": "<cross-section, hole or post, flanges, steps>",
 "mating": "plug_into_socket" | "plug_onto_socket", "action": "<verb phrase, e.g. insert the pin into the hole>",
 "confidence": "high" | "medium" | "low"}"""
def img(p): return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()}}
def ask(t, rep):
    keys = list(VIEWS); random.Random(int(t) * 10 + rep).shuffle(keys)
    content = [{"type": "text", "text": HEAD}]
    for i, k in enumerate(keys):
        content += [{"type": "text", "text": f"Image {i + 1}: {VIEWS[k]}."}, img(f"{D}/{t}_{k}.png")]
    content.append({"type": "text", "text": TAIL})
    body = {"model": MODEL, "temperature": 0, "max_tokens": 700, "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    for attempt in range(8):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=240)); txt = r["choices"][0]["message"]["content"]
            ans = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
            return {"task": t, "rep": rep, "order": keys, "answer": ans, "usage": r.get("usage", {})}
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code} {e.read().decode(errors='ignore')[:200]}"
            if e.code == 429: time.sleep(15 * (attempt + 1))
            elif e.code >= 500: time.sleep(10)
            else: break
        except Exception as e:
            err = str(e); time.sleep(10)
    return {"task": t, "rep": rep, "error": err}
jobs = [(t, rep) for t in sys.argv[3:] for rep in range(3)]
with cf.ThreadPoolExecutor(int(os.environ.get("LLMAPI_WORKERS", "2"))) as ex, open(OUT, "w") as f:
    for res in ex.map(lambda j: ask(*j), jobs):
        f.write(json.dumps(res, ensure_ascii=False) + "\n"); f.flush()
print("done", len(jobs))
