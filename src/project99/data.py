from pathlib import Path

import typer
from torch.utils.data import Dataset

import pandas as pd
import numpy as np

class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path
        self.match_files = sorted(data_path.glob('*-matches.csv'))

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        output_folder.mkdir(parents=True, exist_ok=True)

        for match_file in self.match_files:
            tournament_name = match_file.stem.replace('-matches', '')
            point_file = self.data_path/f'{tournament_name}-points.csv'

            if point_file.exists():
                print(f'Processing tournament: {tournament_name}')
                df_match_raw = pd.read_csv(match_file)
                df_point_raw = pd.read_csv(point_file)

                df_matches_raw_sub = df_match_raw[["match_id", "player1", "player2"]]
                joint_df = pd.merge(df_matches_raw_sub, df_point_raw, on="match_id", how="left")

                # Remove 0 in PointWinner
                joint_df = joint_df[joint_df['PointWinner'].isin([1, 2])].copy()

                need_shift_columns = ['P1GamesWon', 'P2GamesWon', 'P1Score', 'P2Score',
                                    'P1PointsWon', 'P2PointsWon', 'P1Momentum', 'P2Momentum']
                for col in need_shift_columns:
                    joint_df[f'Prev{col}'] = joint_df.groupby('match_id')[col].shift(1).fillna(0)
                
                # Create PrevSetsWon columns
                joint_df['P1SetsWon'] = (joint_df['SetWinner'] == 1).astype(int)
                joint_df['P2SetsWon'] = (joint_df['SetWinner'] == 2).astype(int)

                joint_df['P1SetsWon'] = joint_df.groupby('match_id')['P1SetsWon'].cumsum()
                joint_df['P2SetsWon'] = joint_df.groupby('match_id')['P2SetsWon'].cumsum()

                joint_df['PrevP1SetsWon'] = joint_df.groupby('match_id')['P1SetsWon'].shift(1).fillna(0)
                joint_df['PrevP2SetsWon'] = joint_df.groupby('match_id')['P2SetsWon'].shift(1).fillna(0)

                # Transform features from P1 P2 perspective to Server Receiver perspective
                joint_df[f'Server'] = np.where(joint_df['PointServer'] == 1, joint_df['player1'], joint_df['player2'])
                joint_df[f'Receiver'] = np.where(joint_df['PointServer'] == 1, joint_df['player2'], joint_df['player1'])

                joint_df[f'ServerScore'] = np.where(joint_df['PointServer'] == 1, joint_df['PrevP1Score'], joint_df['PrevP2Score'])
                joint_df[f'ReceiverScore'] = np.where(joint_df['PointServer'] == 1, joint_df['PrevP2Score'], joint_df['PrevP1Score'])

                columns_to_transform = ['GamesWon', 'PointsWon', 'Momentum', 'SetsWon']

                for column in columns_to_transform:
                    joint_df[f'Server{column}'] = np.where(
                        joint_df['PointServer'] == 1,
                        (joint_df[f'PrevP1{column}']).astype(int),
                        (joint_df[f'PrevP2{column}']).astype(int)
                    )
                    joint_df[f'Receiver{column}'] = np.where(
                        joint_df['PointServer'] == 1,
                        (joint_df[f'PrevP2{column}']).astype(int),
                        (joint_df[f'PrevP1{column}']).astype(int)
                    )

                joint_df['ServerWon'] = (joint_df['PointWinner'] == joint_df['PointServer']).astype(int)

                # Remove double fault points
                joint_df = joint_df[joint_df['ServeNumber'].isin([1, 2])].copy()
                
                final_columns = ['match_id', 'SetNo', 'Server', 'Receiver', 'GameNo', 'SetNo', 'ServeNumber',
                                'PointNumber', 'ServerGamesWon', 'ReceiverGamesWon',
                                'ServerScore', 'ReceiverScore', 'ServerPointsWon', 'ReceiverPointsWon',
                                'ServerMomentum', 'ReceiverMomentum', 'ServerSetsWon', 'ReceiverSetsWon', 'ServerWon']
                final_df = joint_df[final_columns].copy()

                output_file = output_folder/f'{tournament_name}-processed.csv'
                final_df.to_csv(output_file, index=False)
            else:
                print(f"Warning: No points file found for {tournament_name}")


def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)
