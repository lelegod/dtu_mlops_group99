from typing import Literal

from pydantic import BaseModel, Field


class RawPointInput(BaseModel):
    SetNo: int = Field(..., ge=1, le=5, description="Current set number (1-5)")
    GameNo: int = Field(..., ge=1, le=20, description="Current game in the set")
    PointNumber: int = Field(..., ge=1, description="Point number in the match")
    PointServer: Literal[1, 2] = Field(..., description="Who is serving: 1=Player1, 2=Player2")
    ServeIndicator: Literal[1, 2] = Field(..., description="1=First serve, 2=Second serve")
    P1GamesWon: int = Field(..., ge=0, le=13, description="Games won by P1 in current set")
    P1SetsWon: int = Field(..., ge=0, le=3, description="Sets won by P1")
    P1Score: Literal["0", "15", "30", "40", "AD"] | int = Field(..., description="P1's score in current game")
    P1PointsWon: int = Field(..., ge=0, description="Total points won by P1 in match")
    P1Momentum: int = Field(default=0, description="P1's momentum score")
    P2GamesWon: int = Field(..., ge=0, le=13, description="Games won by P2 in current set")
    P2SetsWon: int = Field(..., ge=0, le=3, description="Sets won by P2")
    P2Score: Literal["0", "15", "30", "40", "AD"] | int = Field(..., description="P2's score in current game")
    P2PointsWon: int = Field(..., ge=0, description="Total points won by P2 in match")
    P2Momentum: int = Field(default=0, description="P2's momentum score")

    model_config = {
        "json_schema_extra": {
            "examples": [{
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
                "P2Momentum": -1
            }]
        }
    }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0=Receiver wins, 1=Server wins")
    probability: float = Field(..., ge=0, le=1, description="Probability that server wins")


class DetailedPredictionResponse(BaseModel):
    server_win_probability: float = Field(..., ge=0, le=1)
    prediction: Literal["Server Wins", "Receiver Wins"]
    confidence: Literal["High", "Medium", "Low"]
    server_player: Literal["Player 1", "Player 2"]
    processed_features: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str | None = None


class ModelInfoResponse(BaseModel):
    model_type: str
    model_loaded: bool
    model_path: str | None = None
    feature_count: int
    feature_names: list[str]
