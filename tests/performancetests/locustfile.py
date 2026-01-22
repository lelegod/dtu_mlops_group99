from urllib.parse import urlparse

import google.auth
import google.auth.transport.requests
from locust import HttpUser, between, task


class TennisUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        if self.host and "googleapis.com" in self.host:
            import os

            access_token = os.environ.get("ACCESS_TOKEN")
            if access_token:
                self.client.headers["Authorization"] = f"Bearer {access_token}"
                print("Successfully attached Google Cloud credentials from env var.")
                return

            try:
                creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
                auth_req = google.auth.transport.requests.Request()
                creds.refresh(auth_req)
                self.client.headers["Authorization"] = f"Bearer {creds.token}"
                print("Successfully attached Google Cloud credentials.")
            except Exception as e:
                print(f"Warning: Failed to get Google Cloud credentials: {e}")

    @task
    def predict(self):
        # Local API uses /predict
        # Vertex AI uses POST https://.../endpoints/ID:predict

        path = "/predict"
        if self.host and "googleapis.com" in self.host:
            path = ":predict"

        response = self.client.post(
            path,
            json={
                "instances": [
                    {
                        "SetNo": 1,
                        "GameNo": 3,
                        "PointNumber": 15,
                        "PointServer": 1,
                        "ServeIndicator": 1,
                        "P1GamesWon": 2,
                        "P1SetsWon": 0,
                        "P1Score": "30",
                        "P1PointsWon": 12,
                        "P1Momentum": 2,
                        "P2GamesWon": 1,
                        "P2SetsWon": 0,
                        "P2Score": "15",
                        "P2PointsWon": 10,
                        "P2Momentum": -1,
                    }
                ]
            },
        )

        if response.status_code != 200:
            print(f"Request failed with status {response.status_code}: {response.text}")
