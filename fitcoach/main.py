"""FitCoach AI: FastAPI + Groq tool calling + SQLite."""
import os, json, sqlite3, datetime
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from groq import Groq

load_dotenv()
client = Groq()                       # reads GROQ_API_KEY from env
MODEL = "llama-3.3-70b-versatile"     # must support tool use; check console.groq.com/docs/models
DB = "fitcoach.db"
app = FastAPI(title="FitCoach AI")


# ---------------- database ----------------
def db():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


with db() as c:
    c.executescript("""
    CREATE TABLE IF NOT EXISTS profile(user_id TEXT PRIMARY KEY, data TEXT);
    CREATE TABLE IF NOT EXISTS meals(id INTEGER PRIMARY KEY, user_id TEXT, day TEXT,
        name TEXT, calories REAL, protein REAL, carbs REAL, fat REAL);
    CREATE TABLE IF NOT EXISTS workouts(id INTEGER PRIMARY KEY, user_id TEXT, day TEXT,
        description TEXT, minutes INTEGER);
    """)


def get_profile(uid):
    with db() as c:
        r = c.execute("SELECT data FROM profile WHERE user_id=?", (uid,)).fetchone()
    return json.loads(r["data"]) if r else {}


# ---------------- tool functions ----------------
ACTIVITY = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "high": 1.725}


def save_profile(uid, a):
    p = get_profile(uid)
    p.update(a)
    with db() as c:
        c.execute("INSERT OR REPLACE INTO profile VALUES (?,?)", (uid, json.dumps(p)))
    return {"saved": p}


def calculate_targets(uid, a):
    p = get_profile(uid)
    need = ["age", "sex", "weight_kg", "height_cm", "activity", "goal"]
    missing = [k for k in need if k not in p]
    if missing:
        return {"error": f"Missing profile fields: {missing}. Ask the user, then call save_profile."}
    bmr = 10 * p["weight_kg"] + 6.25 * p["height_cm"] - 5 * p["age"] + (5 if p["sex"] == "male" else -161)
    tdee = bmr * ACTIVITY[p["activity"]]
    adj = {"lose": 0.85, "maintain": 1.0, "gain": 1.10}[p["goal"]]
    floor = 1500 if p["sex"] == "male" else 1200      # safety floor
    cal = max(round(tdee * adj), floor)
    protein = round(p["weight_kg"] * (2.0 if p["goal"] == "lose" else 1.8))
    fat = round(cal * 0.25 / 9)
    carbs = round((cal - protein * 4 - fat * 9) / 4)
    return {"bmr": round(bmr), "tdee": round(tdee), "target_calories": cal,
            "protein_g": protein, "carbs_g": carbs, "fat_g": fat}


def log_meal(uid, a):
    with db() as c:
        c.execute("INSERT INTO meals(user_id,day,name,calories,protein,carbs,fat) VALUES (?,?,?,?,?,?,?)",
                  (uid, str(datetime.date.today()), a["name"], a["calories"],
                   a.get("protein", 0), a.get("carbs", 0), a.get("fat", 0)))
    return {"logged": a["name"]}


def log_workout(uid, a):
    with db() as c:
        c.execute("INSERT INTO workouts(user_id,day,description,minutes) VALUES (?,?,?,?)",
                  (uid, str(datetime.date.today()), a["description"], a.get("minutes", 0)))
    return {"logged": a["description"]}


def today_summary(uid, a):
    d = str(datetime.date.today())
    with db() as c:
        m = c.execute("SELECT name,calories,protein,carbs,fat FROM meals WHERE user_id=? AND day=?", (uid, d)).fetchall()
        w = c.execute("SELECT description,minutes FROM workouts WHERE user_id=? AND day=?", (uid, d)).fetchall()
    meals = [dict(r) for r in m]
    tot = {k: round(sum(x[k] for x in meals)) for k in ("calories", "protein", "carbs", "fat")}
    return {"date": d, "meals": meals, "totals": tot,
            "workouts": [dict(r) for r in w], "targets": calculate_targets(uid, {})}


TOOL_FUNCS = {"save_profile": save_profile, "calculate_targets": calculate_targets,
              "log_meal": log_meal, "log_workout": log_workout, "today_summary": today_summary}


# ---------------- tool schemas (what the model sees) ----------------
NUM = {"type": "number"}
TOOL_DEFS = [
    {"name": "save_profile", "description": "Save or update user profile facts (partial updates OK).",
     "input_schema": {"type": "object", "properties": {
         "age": NUM, "sex": {"type": "string", "enum": ["male", "female"]},
         "weight_kg": NUM, "height_cm": NUM,
         "activity": {"type": "string", "enum": list(ACTIVITY)},
         "goal": {"type": "string", "enum": ["lose", "maintain", "gain"]},
         "diet": {"type": "string", "description": "e.g. vegetarian, allergies"},
         "equipment": {"type": "string", "description": "gym, home, dumbbells only"}}}},
    {"name": "calculate_targets", "description": "Compute daily calorie and macro targets from the saved profile.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "log_meal", "description": "Log a meal eaten today. Estimate macros if not given.",
     "input_schema": {"type": "object", "required": ["name", "calories"], "properties": {
         "name": {"type": "string"}, "calories": NUM, "protein": NUM, "carbs": NUM, "fat": NUM}}},
    {"name": "log_workout", "description": "Log a workout done today.",
     "input_schema": {"type": "object", "required": ["description"], "properties": {
         "description": {"type": "string"}, "minutes": NUM}}},
    {"name": "today_summary", "description": "Get today's meals, workouts, totals and targets.",
     "input_schema": {"type": "object", "properties": {}}},
]

# Convert to Groq / OpenAI function format
TOOLS = [{"type": "function",
          "function": {"name": t["name"], "description": t["description"],
                       "parameters": t["input_schema"]}} for t in TOOL_DEFS]

SYSTEM = """You are FitCoach, a friendly evidence-based fitness and nutrition coach.
Rules:
- If profile info is missing, ask briefly, then call save_profile. Never guess body stats.
- ALWAYS use calculate_targets for calorie/macro numbers; never do calorie math yourself.
- ALWAYS call log_meal when the user says they ate something, and log_workout when they trained.
- Workout plans must fit the user's equipment, level and time. Give sets x reps.
- You are not a doctor. For injuries, pregnancy, medical conditions, or signs of
  disordered eating, advise seeing a professional. Never recommend crash diets.
- Keep answers concise and practical. Today is {today}.
Saved profile: {profile}"""


# ---------------- chat endpoint ----------------
HISTORY = {}   # per-user memory in RAM (resets on restart)


class ChatIn(BaseModel):
    user_id: str = "demo"
    message: str


def trim(msgs, keep=30):
    while len(msgs) > keep:
        msgs.pop(0)
    while msgs and msgs[0]["role"] != "user":   # never start on an assistant/tool message
        msgs.pop(0)


@app.post("/chat")
def chat(body: ChatIn):
    uid = body.user_id
    msgs = HISTORY.setdefault(uid, [])
    msgs.append({"role": "user", "content": body.message})
    system = SYSTEM.format(today=datetime.date.today(), profile=json.dumps(get_profile(uid)))
    reply = ""

    for _ in range(6):                                    # max tool rounds
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": system}] + msgs,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.3,
                max_tokens=1500,
            )
        except Exception as e:
            print("Groq error:", e)                       # check your terminal for details
            msgs.pop()
            return {"reply": "Sorry, I hit a hiccup. Please try rephrasing that."}

        m = resp.choices[0].message

        if not m.tool_calls:                              # final answer
            reply = m.content or ""
            msgs.append({"role": "assistant", "content": reply})
            break

        msgs.append({
            "role": "assistant",
            "content": m.content or "",
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name,
                                         "arguments": tc.function.arguments}}
                           for tc in m.tool_calls],
        })
        for tc in m.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
                out = TOOL_FUNCS[tc.function.name](uid, args)
            except Exception as e:
                out = {"error": str(e)}
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out)})
    else:
        reply = "I'm having trouble finishing that. Could you try again?"

    trim(msgs)
    return {"reply": reply}


# ---------------- static file routing & auto-recovery ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

index_path = os.path.join(STATIC_DIR, "index.html")
if not os.path.exists(index_path):
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FitCoach AI</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:640px;margin:0 auto;height:100vh;display:flex;flex-direction:column;background:#f5f7f5}
 h1{font-size:18px;padding:14px 16px;margin:0;background:#1f7a4d;color:#fff}
 #log{flex:1;overflow-y:auto;padding:16px}
 .m{max-width:85%;padding:10px 14px;border-radius:14px;margin:8px 0;white-space:pre-wrap;line-height:1.45}
 .u{background:#1f7a4d;color:#fff;margin-left:auto}.a{background:#fff;border:1px solid #dde}
 form{display:flex;gap:8px;padding:12px;background:#fff;border-top:1px solid #dde}
 input{flex:1;padding:12px;border:1px solid #ccd;border-radius:10px;font-size:16px}
 button{padding:0 18px;border:0;border-radius:10px;background:#1f7a4d;color:#fff;font-size:16px}
</style></head><body>
<h1>🏋️ FitCoach AI</h1><div id="log"></div>
<form id="f"><input id="t" placeholder="e.g. I'm 28, 70kg, 175cm, want to lose fat" autocomplete="off"><button>Send</button></form>
<script>
const log = document.getElementById('log'), t = document.getElementById('t');
const uid = localStorage.uid || (localStorage.uid = crypto.randomUUID());
function add(c, x) {
  const d = document.createElement('div');
  d.className = 'm ' + c; d.textContent = x;
  log.append(d); log.scrollTop = log.scrollHeight; return d;
}
add('a', "Hi! I'm your fitness & nutrition coach. Tell me about yourself and your goal.");
document.getElementById('f').onsubmit = async e => {
  e.preventDefault();
  const msg = t.value.trim(); if (!msg) return;
  t.value = ''; add('u', msg);
  const w = add('a', '…');
  try {
    const r = await fetch('/chat', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                    body: JSON.stringify({user_id: uid, message: msg})});
    w.textContent = (await r.json()).reply || 'Something went wrong.';
  } catch { w.textContent = 'Network error.'; }
};
</script></body></html>""")

@app.get("/")
def home():
    return FileResponse(index_path)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")