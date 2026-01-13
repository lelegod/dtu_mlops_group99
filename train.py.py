import pandas as pd 


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import hydra 
from omegaconf import DictConfig, OmegaConf
import os

@hydra.main(version_base=None,config_path=".", config_name="config")
def main(cfg: DictConfig):
      print("Started")
      df = pd.read_csv(r"C:\Users\akash\Downloads\train_set.csv")
      df.head()
      raw_data = df.to_numpy()
      headers = df.columns.tolist()
      xx = {name: i for i, name in enumerate(headers)}


      game_diff = raw_data[:, xx['ServerGamesWon']] - raw_data[:, xx['ReceiverGamesWon']]
      set_diff = raw_data[:, xx['ServerSetsWon']] - raw_data[:, xx['ReceiverSetsWon']]
      point_diff = raw_data[:, xx['ServerPointsWon']] - raw_data[:, xx['ReceiverPointsWon']]
      mom_diff = raw_data[:, xx['ServerMomentum']] - raw_data[:, xx['ReceiverMomentum']]


      is_break_point = np.logical_and(
      raw_data[:, xx['ReceiverScore']] >= 3,
      raw_data[:, xx['ReceiverScore']] > raw_data[:, xx['ServerScore']]).astype(int)


      max_games = np.maximum(raw_data[:, xx['ServerGamesWon']], raw_data[:, xx['ReceiverGamesWon']])
      set_pressure = 6 - max_games


      is_second_serve = (raw_data[:, xx['ServeIndicator']] == 2).astype(int)


      keep_cols = ['SetNo', 'GameNo', 'PointNumber', 'ServerGamesWon', 'ReceiverGamesWon', 'ServerScore', 'ReceiverScore', 
     'ServerMomentum', 'ReceiverMomentum', 'ServerSetsWon', 'ReceiverSetsWon']
 
      keep_indices = [xx[col] for col in keep_cols]
      base_features = raw_data[:, keep_indices]


      X = np.column_stack((base_features, game_diff, set_diff, point_diff, mom_diff, is_break_point, set_pressure, is_second_serve))

      y = raw_data[:, xx['ServerWon']]


      X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


      print("Training XGBoost Model...")


      model = xgb.XGBClassifier(
      n_estimators=500,learning_rate=0.05,max_depth=6,subsample=0.8,colsample_bytree=0.8,objective='binary:logistic',eval_metric='logloss',early_stopping_rounds=50,random_state=42,n_jobs=-1)


      model.fit(X_train, y_train,eval_set=[(X_val, y_val)],verbose=True)
      y_prob = model.predict_proba(X_val)[:, 1]
      y_pred = model.predict(X_val)

      print("\n--- Model Performance ---")
      print(f"Log Loss:    {log_loss(y_val, y_prob):.4f}")
      print(f"Brier Score: {brier_score_loss(y_val, y_prob):.4f}")
      print(f"AUC Score:   {roc_auc_score(y_val, y_prob):.4f}")
      print(f"Accuracy:    {accuracy_score(y_val, y_pred):.4f}")

if __name__ == "__main__":
    main()