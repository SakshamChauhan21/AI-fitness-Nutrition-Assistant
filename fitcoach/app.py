"""FitCoach AI: Streamlit + Groq tool calling + SQLite."""
import os
import json
import sqlite3
import datetime
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Page configuration
st.set_page_config(page_title="FitCoach AI", page_icon="🏋️", layout="centered")

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "qwen/qwen3.8-27b"             # Updated to supported Qwen model on Groq
DB = "fitcoach.db"

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
    floor = 1500 if p["sex"] == "male" else 1200
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

TOOL_FUNCS = {
    "save_profile": save_profile,
    "calculate_targets": calculate_targets,
    "log_meal": log_meal,
    "log_workout": log_workout,
    "today_summary": today_summary
}

# ---------------- tool schemas ----------------
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

TOOLS = [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}} for t in TOOL_DEFS]

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

# ---------------- Streamlit UI ----------------
st.title("🏋️ FitCoach AI")

# Persistent user ID state
if "user_id" not in st.session_state:
    st.session_state.user_id = "default_user"

# Chat history state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your fitness & nutrition coach. Tell me about yourself and your goal."}
    ]

# Display existing messages
for msg in st.session_state.messages:
    if msg["role"] in ["user", "assistant"] and msg.get("content"):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# User input box
if prompt := st.chat_input("Tell me about your goals, meals, or workouts..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare system prompt with current user profile
    profile_data = get_profile(st.session_state.user_id)
    system_prompt = SYSTEM.format(today=datetime.date.today(), profile=json.dumps(profile_data))

    api_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages if m["role"] in ["user", "assistant"]]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = ""
            # Tool calling loop
            for _ in range(6):
                try:
                    resp = client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "system", "content": system_prompt}] + api_messages,
                        tools=TOOLS,
                        tool_choice="auto",
                        temperature=0.3,
                        max_tokens=1500,
                    )
                except Exception as e:
                    reply = f"Sorry, I hit an error: {e}"
                    break

                m = resp.choices[0].message

                if not m.tool_calls:
                    reply = m.content or ""
                    break

                api_messages.append({
                    "role": "assistant",
                    "content": m.content or "",
                    "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in m.tool_calls]
                })

                for tc in m.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                        out = TOOL_FUNCS[tc.function.name](st.session_state.user_id, args)
                    except Exception as e:
                        out = {"error": str(e)}
                    api_messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out)})
            
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})