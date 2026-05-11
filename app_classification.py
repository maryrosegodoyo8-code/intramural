import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load('sports_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

st.title('Sports Match Outcome Predictor')

# User Inputs
match_id = st.text_input('Match ID', 'M101')
sport = st.selectbox('Sport', label_encoders['sport'].classes_)
team_home = st.selectbox('Home Team', label_encoders['team_home'].classes_)
team_away = st.selectbox('Away Team', label_encoders['team_away'].classes_)
venue = st.selectbox('Venue', label_encoders['venue'].classes_)
day_of_week = st.selectbox('Day of Week', label_encoders['day_of_week'].classes_)
time_slot = st.selectbox('Time Slot', label_encoders['time_slot'].classes_)

home_team_rank = st.number_input('Home Team Rank', 1, 20)
away_team_rank = st.number_input('Away Team Rank', 1, 20)
home_prev_wins = st.number_input('Home Previous Wins', 0, 20)
away_prev_wins = st.number_input('Away Previous Wins', 0, 20)

home_avg_score = st.number_input('Home Average Score', 0.0, 100.0)
away_avg_score = st.number_input('Away Average Score', 0.0, 100.0)

home_attendance_rate = st.number_input('Home Attendance Rate', 0.0, 1.0)
away_attendance_rate = st.number_input('Away Attendance Rate', 0.0, 1.0)

home_fatigue_index = st.number_input('Home Fatigue Index', 0.0, 1.0)
away_fatigue_index = st.number_input('Away Fatigue Index', 0.0, 1.0)

schedule_conflict_count = st.number_input('Schedule Conflict Count', 0, 10)

days_since_last_match_home = st.number_input('Days Since Last Match Home', 0, 30)
days_since_last_match_away = st.number_input('Days Since Last Match Away', 0, 30)

referee_experience_years = st.number_input('Referee Experience Years', 0, 30)

weather_condition = st.selectbox(
    'Weather Condition',
    label_encoders['weather_condition'].classes_
)

audience_count = st.number_input('Audience Count', 0, 10000)
performance_score = st.number_input('Performance Score', 0.0, 100.0)

# Encode inputs
input_data = {
    'match_id': label_encoders['match_id'].transform([match_id])[0]
    if match_id in label_encoders['match_id'].classes_
    else 0,

    'sport': label_encoders['sport'].transform([sport])[0],
    'team_home': label_encoders['team_home'].transform([team_home])[0],
    'team_away': label_encoders['team_away'].transform([team_away])[0],
    'venue': label_encoders['venue'].transform([venue])[0],
    'day_of_week': label_encoders['day_of_week'].transform([day_of_week])[0],
    'time_slot': label_encoders['time_slot'].transform([time_slot])[0],

    'home_team_rank': home_team_rank,
    'away_team_rank': away_team_rank,
    'home_prev_wins': home_prev_wins,
    'away_prev_wins': away_prev_wins,
    'home_avg_score': home_avg_score,
    'away_avg_score': away_avg_score,
    'home_attendance_rate': home_attendance_rate,
    'away_attendance_rate': away_attendance_rate,
    'home_fatigue_index': home_fatigue_index,
    'away_fatigue_index': away_fatigue_index,
    'schedule_conflict_count': schedule_conflict_count,
    'days_since_last_match_home': days_since_last_match_home,
    'days_since_last_match_away': days_since_last_match_away,
    'referee_experience_years': referee_experience_years,
    'weather_condition': label_encoders['weather_condition'].transform([weather_condition])[0],
    'audience_count': audience_count,
    'performance_score': performance_score
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Prediction
if st.button('Predict Match Outcome'):
    prediction = model.predict(input_df)

    result = label_encoders['match_outcome'].inverse_transform(prediction)

    st.success(f'Predicted Match Outcome: {result[0]}')