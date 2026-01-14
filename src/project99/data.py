from pathlib import Path

import numpy as np
import pandas as pd
import torch
import typer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset


class TennisDataProcessor:
    """Tennis data processor for preprocessing raw dataset."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path
        self.match_files = sorted(data_path.glob('*-matches.csv'))

    def _load_tournament_data(self, match_file: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
        """Load matches and points dataset from a tournament file path"""
        tournament_name = match_file.stem.replace('-matches', '')
        point_file = self.data_path/f'{tournament_name}-points.csv'

        if not point_file.exists():
            raise FileNotFoundError(f'Points file not found for tournament: {tournament_name}')

        df_match_raw = pd.read_csv(match_file)
        df_point_raw = pd.read_csv(point_file)

        return df_match_raw, df_point_raw, tournament_name

    def _create_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features for features that need previous point information."""
        need_shift_columns =['P1GamesWon', 'P2GamesWon', 'P1Score', 'P2Score',
                            'P1PointsWon', 'P2PointsWon', 'P1Momentum', 'P2Momentum']

        for col in need_shift_columns:
            df[f'Prev{col}'] = df.groupby('match_id')[col].shift(1).fillna(0)

        return df

    def _create_sets_won(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create SetsWon before the point starts."""
        df['P1SetsWon'] = (df['SetWinner'] == 1).astype(int)
        df['P2SetsWon'] = (df['SetWinner'] == 2).astype(int)

        df['P1SetsWon'] = df.groupby('match_id')['P1SetsWon'].cumsum()
        df['P2SetsWon'] = df.groupby('match_id')['P2SetsWon'].cumsum()

        df['PrevP1SetsWon'] = df.groupby('match_id')['P1SetsWon'].shift(1).fillna(0)
        df['PrevP2SetsWon'] = df.groupby('match_id')['P2SetsWon'].shift(1).fillna(0)

        return df

    def _transform_to_server_receiver(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features from P1/P2 perspective to Server/Receiver perspective."""
        df[f'Server'] = np.where(df['PointServer'] == 1, df['player1'], df['player2'])
        df[f'Receiver'] = np.where(df['PointServer'] == 1, df['player2'], df['player1'])

        df[f'ServerScore'] = np.where(df['PointServer'] == 1, df['PrevP1Score'], df['PrevP2Score'])
        df[f'ReceiverScore'] = np.where(df['PointServer'] == 1, df['PrevP2Score'], df['PrevP1Score'])

        # Handle regular and tiebreak score mapping
        def map_score(score: str | int) -> int:
            regular_score_map = {'0': 0, '15': 1, '30': 2, '40': 3, 'AD': 4,
                                0: 0, 15: 1, 30: 2, 40: 3}
            if score in regular_score_map:
                return regular_score_map[score]
            try:
                return int(score)
            except Exception as e:
                return np.nan

        df['ServerScore'] = df['ServerScore'].apply(map_score)
        df['ReceiverScore'] = df['ReceiverScore'].apply(map_score)

        columns_to_transform = ['GamesWon', 'PointsWon', 'Momentum', 'SetsWon']

        for column in columns_to_transform:
            df[f'Server{column}'] = np.where(
                df['PointServer'] == 1,
                (df[f'PrevP1{column}']).astype(int),
                (df[f'PrevP2{column}']).astype(int)
            )
            df[f'Receiver{column}'] = np.where(
                df['PointServer'] == 1,
                (df[f'PrevP2{column}']).astype(int),
                (df[f'PrevP1{column}']).astype(int)
            )

        df['ServerWon'] = (df['PointWinner'] == df['PointServer']).astype(int)

        return df

    def _create_score_differentials(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create score differential features."""
        df['GameDiff'] = df['ServerGamesWon'] - df['ReceiverGamesWon']
        df['SetDiff'] = df['ServerSetsWon'] - df['ReceiverSetsWon']
        df['PointDiff'] = df['ServerPointsWon'] - df['ReceiverPointsWon']
        df['MomentumDiff'] = df['ServerMomentum'] - df['ReceiverMomentum']
        return df

    def _engineer_contextual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create contextual features like break points and set pressure."""
        # Break point situation (receiver has advantage)
        df['IsBreakPoint'] = ((df['ReceiverScore'] >= 3) &
                                (df['ReceiverScore'] > df['ServerScore'])).astype(int)
        
        # Set pressure (proximity to winning the set at 6 games)
        max_games = np.maximum(df['ServerGamesWon'], df['ReceiverGamesWon'])
        df['SetPressure'] = 6 - max_games
        
        df['IsSecondServe'] = (df['ServeIndicator'] == 2).astype(int)
        
        return df

    def _process_single_tournament(self, match_file: Path) -> tuple[pd.DataFrame, str]:
        """Process a single tournament's data"""
        df_match_raw, df_point_raw, tournament_name = self._load_tournament_data(match_file)

        # Merge matches and points dataset
        df_matches_raw_sub = df_match_raw[["match_id", "player1", "player2"]]
        joint_df = pd.merge(df_matches_raw_sub, df_point_raw, on="match_id", how="left")

        # Remove 0 in PointWinner
        joint_df = joint_df[joint_df['PointWinner'].isin([1, 2])].copy()

        # Create lagged features
        joint_df = self._create_lagged_features(joint_df)

        # Create SetsWon features
        joint_df = self._create_sets_won(joint_df)

        # Transform to Server/Receiver perspective
        joint_df = self._transform_to_server_receiver(joint_df)

        # Remove double fault points
        joint_df = joint_df[joint_df['ServeIndicator'].isin([1, 2])].copy()

        # Create score differentials
        joint_df = self._create_score_differentials(joint_df)

        # Create contextual features
        joint_df = self._engineer_contextual_features(joint_df)

        # Note: 'Server' and 'Receiver' are excluded for now
        final_columns = ['SetNo', 'GameNo', 'PointNumber', 'ServerGamesWon', 'ReceiverGamesWon',
                        'ServerScore', 'ReceiverScore', 'ServerPointsWon', 'ReceiverPointsWon',
                        'ServerMomentum', 'ReceiverMomentum', 'ServerSetsWon', 'ReceiverSetsWon',
                        'GameDiff', 'SetDiff', 'PointDiff', 'MomentumDiff',
                        'IsBreakPoint', 'SetPressure', 'IsSecondServe', 'ServerWon']

        # Remove NaN rows if any
        final_df = joint_df[final_columns].dropna()

        return final_df, tournament_name

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        output_folder.mkdir(parents=True, exist_ok=True)

        # Store all processed dataframes for merging
        all_processed_dfs: list[pd.DataFrame] = []

        for match_file in self.match_files:
            try:
                processed_df, tournament_name = self._process_single_tournament(match_file)
                print(f'Processed and saved tournament: {tournament_name}')

                # Append to list for merging
                all_processed_dfs.append(processed_df)

            except Exception as e:
                print(f"Error processing {match_file.stem}: {e}")
                continue

        if not all_processed_dfs:
            raise ValueError("No tournaments were processed successfully.")

        # Merge all processed dataframes
        merged_df = pd.concat(all_processed_dfs, ignore_index=True)

        # Split train test set
        X = merged_df.drop('ServerWon', axis=1).values
        y = merged_df['ServerWon'].values

        # Stratified train-test split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # Save merged train and test sets
        train_df = pd.DataFrame(X_train, columns=merged_df.columns[:-1])
        train_df['ServerWon'] = y_train

        test_df = pd.DataFrame(X_test, columns=merged_df.columns[:-1])
        test_df['ServerWon'] = y_test

        train_file = output_folder/'train_set.csv'
        train_df.to_csv(train_file, index=False)
        print(f'Saved training set to {train_file}')
        test_file = output_folder/'test_set.csv'
        test_df.to_csv(test_file, index=False)
        print(f'Saved test set to {test_file}')


def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = TennisDataProcessor(data_path)
    dataset.preprocess(output_folder)


def tennis_data(
        data_type: str = 'torch'
        ) -> tuple[TensorDataset, TensorDataset] |tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Load the tennis train and test data

    Args:
        data_type: 'torch' for PyTorch tensors or 'numpy' for NumPy arrays
    """
    train_file = Path('data/processed/train_set.csv')
    test_file = Path('data/processed/test_set.csv')

    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(f'Processed data files not found.')

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    X_train = train_df.drop('ServerWon', axis=1).values
    y_train = train_df['ServerWon'].values
    X_test = test_df.drop('ServerWon', axis=1).values
    y_test = test_df['ServerWon'].values

    if data_type == 'torch':
        train_set = torch.tensor(X_train, dtype=torch.float32)
        train_target = torch.tensor(y_train, dtype=torch.long)
        test_set = torch.tensor(X_test, dtype=torch.float32)
        test_target = torch.tensor(y_test, dtype=torch.long)

        train_dataset = TensorDataset(train_set, train_target)
        test_dataset = TensorDataset(test_set, test_target)

        return train_dataset, test_dataset
    elif data_type == 'numpy':
        return (X_train, y_train), (X_test, y_test)
    else:
        raise ValueError(f'Unsupported data_type: {data_type}. Choose "torch" or "numpy".')


if __name__ == "__main__":
    typer.run(preprocess)
