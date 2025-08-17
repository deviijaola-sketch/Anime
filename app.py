from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os, json, re
from openai import OpenAI

app = FastAPI()

# allow cross-origin calls (so your frontend can talk to this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# client with your OpenAI API key from Render environment
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# simple home route
@app.get("/")
def home():
    return {"ok": True, "service": "anime-gpt-server"}

# helper: safer JSON parse
def safe_json_parse(s: str):
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", s)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None

# main endpoint: get anime titles
@app.post("/titles")
def titles(body: dict):
    text = (body.get("text") or "").strip()
    mood = (body.get("mood") or "").strip()
    raw_max = body.get("max", 2)
    try:
        raw_max = int(raw_max)
    except Exception:
        raw_max = 2
    max_n = max(2, min(3, raw_max))  # clamp 2–3

    if not text and not mood:
        raise HTTPException(status_code=400, detail="text or mood required")

    system = f"""
You are an anime recommender. Return ONLY strict JSON with exact anime titles.
Format: {{"titles":["Exact Title 1","Exact Title 2"]}}
Rules:
- Return between 2 and {max_n} titles (inclusive).
- Prefer official English titles; if missing, use Romaji.
- Match the user's description and mood closely.
- No commentary. No extra keys. Strict JSON only.
""".strip()

    user = f"Description: {text}\nMood: {mood or '(none)'}\nNeed: {max_n} titles. Return JSON now."

    r = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}]
    )
    raw = (r.choices[0].message.content or "").strip()
    data = safe_json_parse(raw)

    if not data or not isinstance(data.get("titles"), list):
        # retry once
        r2 = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user},
                      {"role": "user", "content": "Return strict JSON only as specified."}]
        )
        raw = (r2.choices[0].message.content or "").strip()
        data = safe_json_parse(raw)

    if not data or not isinstance(data.get("titles"), list):
        raise HTTPException(status_code=502, detail="bad_ai_json")

    # clean + dedupe + clamp
    seen = set()
    titles = []
    for t in data["titles"]:
        t = (t or "").strip()
        if t and t not in seen:
            seen.add(t)
            titles.append(t)

    if len(titles) < 2:
        raise HTTPException(status_code=502, detail="need_at_least_two_titles")
    if len(titles) > max_n:
        titles = titles[:max_n]

    return {"titles": titles}
