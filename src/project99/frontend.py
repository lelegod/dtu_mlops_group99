import os

import google.auth  # type: ignore
import google.auth.transport.requests  # type: ignore
import requests
import streamlit as st  # type: ignore

# Configuration
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")


def get_prediction(point_data: dict) -> dict | None:
    payload = {"instances": [point_data]}

    headers = {}
    is_vertex = "googleapis.com" in BACKEND_URL
    url = str(BACKEND_URL)

    if is_vertex:
        try:
            creds, project = google.auth.default()
            auth_req = google.auth.transport.requests.Request()
            creds.refresh(auth_req)
            headers["Authorization"] = f"Bearer {creds.token}"
        except Exception as e:
            st.error(f"Auth failed: {e}")
            return None

        if not url.endswith(":predict"):
            url = f"{url}:predict"
    else:
        if not url.endswith("/predict"):
            url = f"{url}/predict"

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()

        resp_json = response.json()
        if "predictions" in resp_json:
            return resp_json["predictions"][0]
        return resp_json

    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to backend at {url}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error: {e}")
        return None


def main():
    st.set_page_config(page_title="Tennis Point Predictor", page_icon="🎾", layout="wide")

    st.markdown(
        """
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            color: #2E7D32;
            margin-bottom: 1rem;
        }
        .sub-header {
            text-align: center;
            color: #666;
            margin-bottom: 2rem;
        }
        .prediction-box {
            padding: 2rem;
            border-radius: 1rem;
            text-align: center;
            margin: 1rem 0;
        }
        .server-wins {
            background: linear-gradient(135deg, #4CAF50, #2E7D32);
            color: white;
        }
        .receiver-wins {
            background: linear-gradient(135deg, #f44336, #c62828);
            color: white;
        }
        .stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #2E7D32, #4CAF50);
            color: white;
            font-size: 1.2rem;
            padding: 0.75rem;
            border: none;
            border-radius: 0.5rem;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #1B5E20, #2E7D32);
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="main-header">🎾 Tennis Point Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict whether the server will win the point</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Match State")

        set_no = st.number_input("Set Number", min_value=1, max_value=5, value=1)
        game_no = st.number_input("Game Number", min_value=1, max_value=20, value=3)
        point_number = st.number_input("Point Number in Match", min_value=1, value=15)

        st.subheader("🎯 Serve Info")
        point_server = st.radio("Who is Serving?", options=[1, 2], format_func=lambda x: f"Player {x}")
        serve_indicator = st.radio(
            "Serve Type", options=[1, 2], format_func=lambda x: "First Serve" if x == 1 else "Second Serve"
        )

    with col2:
        st.subheader("👤 Player 1")
        p1_games_won = st.number_input("P1 Games Won (this set)", min_value=0, max_value=13, value=2)
        p1_sets_won = st.number_input("P1 Sets Won", min_value=0, max_value=3, value=0)
        p1_score = st.selectbox("P1 Score (this game)", options=["0", "15", "30", "40", "AD"], index=2)
        p1_points_won = st.number_input("P1 Total Points Won", min_value=0, value=12)
        p1_momentum = st.slider("P1 Momentum", min_value=-10, max_value=10, value=2)

        st.subheader("👤 Player 2")
        p2_games_won = st.number_input("P2 Games Won (this set)", min_value=0, max_value=13, value=1)
        p2_sets_won = st.number_input("P2 Sets Won", min_value=0, max_value=3, value=0)
        p2_score = st.selectbox("P2 Score (this game)", options=["0", "15", "30", "40", "AD"], index=1)
        p2_points_won = st.number_input("P2 Total Points Won", min_value=0, value=10)
        p2_momentum = st.slider("P2 Momentum", min_value=-10, max_value=10, value=-1)

    st.divider()

    if st.button("🎾 Predict Point Outcome", use_container_width=True):
        # Prepare the data
        point_data = {
            "SetNo": set_no,
            "GameNo": game_no,
            "PointNumber": point_number,
            "PointServer": point_server,
            "ServeIndicator": serve_indicator,
            "P1GamesWon": p1_games_won,
            "P1SetsWon": p1_sets_won,
            "P1Score": p1_score,
            "P1PointsWon": p1_points_won,
            "P1Momentum": p1_momentum,
            "P2GamesWon": p2_games_won,
            "P2SetsWon": p2_sets_won,
            "P2Score": p2_score,
            "P2PointsWon": p2_points_won,
            "P2Momentum": p2_momentum,
        }

        with st.spinner("Predicting..."):
            result = get_prediction(point_data)

        if result:
            prediction = result.get("prediction", 0)
            probability = result.get("probability", 0.5)

            st.subheader("📈 Prediction Result")

            col_result1, col_result2, col_result3 = st.columns([1, 2, 1])

            with col_result2:
                if prediction == 1:
                    st.success(f"🎾 **SERVER WINS** (Player {point_server})")
                    st.metric("Confidence", f"{probability * 100:.1f}%")
                else:
                    receiver = 2 if point_server == 1 else 1
                    st.error(f"🎾 **RECEIVER WINS** (Player {receiver})")
                    st.metric("Confidence", f"{(1 - probability) * 100:.1f}%")

                st.progress(probability)
                st.caption(f"Server win probability: {probability * 100:.1f}%")

    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This app predicts the outcome of a tennis point based on the current match state.

        **Model**: XGBoost Classifier
        **Features**: 20 engineered features
        **Training Data**: Professional tennis matches

        ---

        **How to use:**
        1. Enter the current match state
        2. Set player scores and statistics
        3. Click "Predict" to see the outcome

        ---

        **Backend Status:**
        """)

        try:
            health = requests.get(f"{BACKEND_URL}/health", timeout=5)
            if health.status_code == 200:
                st.success("✅ Backend connected")
            else:
                st.warning("⚠️ Backend responding but unhealthy")
        except Exception:
            st.error("❌ Backend not connected")


if __name__ == "__main__":
    main()
