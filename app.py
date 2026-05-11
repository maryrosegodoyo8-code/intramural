import streamlit as st
import joblib
import pandas as pd

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(page_title="Sports Performance Predictor")

st.title("🏆 Intramural Sports Performance Predictor")
st.write("Enter match information to predict performance score.")

# =========================================
# LOAD MODEL (CACHED = FAST)
# =========================================
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# =========================================
# INPUT FIELDS
# =========================================
home_team_rank = st.number_input("Home Team Rank", 1, 20, 1)
away_team_rank = st.number_input("Away Team Rank", 1, 20, 1)

home_prev_wins = st.number_input("Home Previous Wins", 0, value=0)
away_prev_wins = st.number_input("Away Previous Wins", 0, value=0)

home_avg_score = st.number_input("Home Average Score", value=0.0)
away_avg_score = st.number_input("Away Average Score", value=0.0)

home_attendance_rate = st.number_input("Home Attendance Rate", 0.0, 1.0, 0.5)
away_attendance_rate = st.number_input("Away Attendance Rate", 0.0, 1.0, 0.5)

home_fatigue_index = st.number_input("Home Fatigue Index", 0.0, 1.0, 0.5)
away_fatigue_index = st.number_input("Away Fatigue Index", 0.0, 1.0, 0.5)

schedule_conflict_count = st.number_input("Schedule Conflict Count", 0, value=0)

days_since_last_match_home = st.number_input("Days Since Last Match (Home)", 0, value=1)
days_since_last_match_away = st.number_input("Days Since Last Match (Away)", 0, value=1)

referee_experience_years = st.number_input("Referee Experience Years", 0, value=1)

audience_count = st.number_input("Audience Count", 0, value=0)

# =========================================
# PREDICTION
# =========================================
if st.button("Predict Performance Score"):

    features = pd.DataFrame([{
        "home_team_rank": home_team_rank,
        "away_team_rank": away_team_rank,
        "home_prev_wins": home_prev_wins,
        "away_prev_wins": away_prev_wins,
        "home_avg_score": home_avg_score,
        "away_avg_score": away_avg_score,
        "home_attendance_rate": home_attendance_rate,
        "away_attendance_rate": away_attendance_rate,
        "home_fatigue_index": home_fatigue_index,
        "away_fatigue_index": away_fatigue_index,
        "schedule_conflict_count": schedule_conflict_count,
        "days_since_last_match_home": days_since_last_match_home,
        "days_since_last_match_away": days_since_last_match_away,
        "referee_experience_years": referee_experience_years,
        "audience_count": audience_count
    }])

    prediction = model.predict(features)[0]

    st.success(f"Predicted Performance Score: {prediction:.2f}")