import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "https://dtu-mlops-api-487762099723.europe-west1.run.app")

st.title("🎾 Tennis Point Predictor")


col1, col2 = st.columns(2)

with col1:
    st.subheader("Match Context")
    set_no = st.number_input("Set Number", min_value=1, max_value=5, value=1)
    game_no = st.number_input("Game Number", min_value=1, max_value=20, value=1)
    point_no = st.number_input("Point Number", min_value=1, value=1)
    server = st.selectbox("Server", options=[1, 2], help="1 for Player 1, 2 for Player 2")
    serve_ind = st.selectbox("Serve Indicator", options=[1, 2], help="1 for First Serve, 2 for Second Serve")

with col2:
    st.subheader("Current Score")
    p1_sets = st.number_input("P1 Sets Won", min_value=0, max_value=3, value=0)
    p1_games = st.number_input("P1 Games Won", min_value=0, max_value=7, value=0)
    p1_score = st.text_input("P1 Score (e.g., 0, 15, 30, 40, AD)", value="0")
    p1_points = st.number_input("P1 Points Won", min_value=0, value=0)
    
    st.divider()
    
    p2_sets = st.number_input("P2 Sets Won", min_value=0, max_value=3, value=0)
    p2_games = st.number_input("P2 Games Won", min_value=0, max_value=7, value=0)
    p2_score = st.text_input("P2 Score (e.g., 0, 15, 30, 40, AD)", value="0")
    p2_points = st.number_input("P2 Points Won", min_value=0, value=0)

if st.button("Predict Point Outcome"):

    payload = {
        "SetNo": set_no,
        "GameNo": game_no,
        "PointNumber": point_no,
        "PointServer": server,
        "ServeIndicator": serve_ind,
        "P1GamesWon": p1_games,
        "P1SetsWon": p1_sets,
        "P1Score": p1_score,
        "P1PointsWon": p1_points,
        "P2GamesWon": p2_games,
        "P2SetsWon": p2_sets,
        "P2Score": p2_score,
        "P2PointsWon": p2_points
    }
    
    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        if response.status_code == 200:
            res = response.json()
            st.success(f"Prediction: {'Player 1 Wins' if res['prediction'] == 1 else 'Player 2 Wins'}")
            st.metric("Confidence", f"{res['probability']*100:.1f}%")
        else:
            st.error(f"Error: {response.text}")
    except Exception as e:
        st.error(f"Connection failed: {e}")