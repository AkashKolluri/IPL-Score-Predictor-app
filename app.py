# 🔧 Prevent inotify crash on Streamlit Cloud
import os
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessor
rf = joblib.load('rf_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

st.title('🏏 IPL Score Predictor')

# Dropdown options
stadiums = [
    "M Chinnaswamy Stadium", "Wankhede Stadium",
    "Eden Gardens", "Arun Jaitley Stadium",
    "MA Chidambaram Stadium", "Rajiv Gandhi International Stadium"
]

teams = [
    "Kolkata Knight Riders", "Royal Challengers Bangalore",
    "Mumbai Indians", "Chennai Super Kings",
    "Rajasthan Royals", "Delhi Capitals",
    "Punjab Kings", "Sunrisers Hyderabad"
]

# Inputs
col1, col2 = st.columns(2)
with col1:
    venue = st.selectbox("Venue", options=stadiums)
    bat_team = st.selectbox("Batting Team", options=teams)
with col2:
    bowl_team = st.selectbox("Bowling Team", options=teams)
    completed_overs = st.number_input("Completed Overs", min_value=5, max_value=19, step=1)
    balls = st.number_input("Balls in Current Over", min_value=0, max_value=5, step=1)

overs = completed_overs + balls / 6

runs = st.number_input("Current Runs", min_value=0)
wickets = st.number_input("Wickets Fallen", 0, 10)
runs_last_5 = st.number_input("Runs in Last 5 Overs", min_value=0)
wickets_last_5 = st.number_input("Wickets in Last 5 Overs", 0, 10)

if st.button("Predict Score"):
    input_df = pd.DataFrame([[venue, bat_team, bowl_team, runs, wickets, overs, runs_last_5, wickets_last_5]],
        columns=['venue', 'bat_team', 'bowl_team', 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5']
    )

    processed_input = preprocessor.transform(input_df)
    prediction = rf.predict(processed_input)[0]
    st.success(f"## Predicted Final Score: {int(prediction)}")
    st.balloons()
