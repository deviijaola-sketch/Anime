from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os, json, re, traceback
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# Read key once. Don't crash if missing; we'll check at request time.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

@app.get("/")
@app.get("/health")
def health():
    return {"ok": True, "service": "anime-gpt-server"}

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

@app.post("/titles")
def titles(body: dict):
    # 0) validate env/key
    if client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set on server")

    # 1) read inputs
    text = (body.get("text") or "").strip()
    mood = (body.get("mood") or "").strip()
    try:
        raw_max = int(body.get("max", 3))
    except Exception:
        raw_max = 3
    max_n = max(2, min(3, raw_max))  # clamp 2..3

    if not text and not mood:
        raise HTTPException(status_code=400, detail="text or mood required")

    # 2) build prompt
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

    # 3) call OpenAI with clear error handling
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        raw = (r.choices[0].message.content or "").strip()
        data = safe_json_parse(raw)

        if not data or not isinstance(data.get("titles"), list):
            # retry once
            r2 = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    {"role": "user", "content": "Return strict JSON only as specified."},
                ],
            )
            raw = (r2.choices[0].message.content or "").strip()
            data = safe_json_parse(raw)

    except Exception as e:
        # print full stack to server logs and return a clean error to client
        traceback.print_exc()
        msg = str(e)
        # Surface common issues in a friendly way
        if "insufficient_quota" in msg or "billing" in msg:
            raise HTTPException(status_code=502, detail="OpenAI: insufficient quota/billing")
        if "invalid_api_key" in msg or "Incorrect API key" in msg:
            raise HTTPException(status_code=401, detail="OpenAI: invalid API key")
        raise HTTPException(status_code=502, detail=f"OpenAI error: {msg}")

    if not data or not isinstance(data.get("titles"), list):
        raise HTTPException(status_code=502, detail="bad_ai_json")

    # 4) clean results
    seen, titles = set(), []
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
