import os
import pandas as pd
import json
from google.cloud import storage
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from project99.constants import GCS_MODEL_PATH

def generate_drift_report():
    client = storage.Client()
    bucket_name = GCS_MODEL_PATH[5:].split("/", 1)[0]
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix="monitoring/prediction_logs/")
    
    logs = []
    for blob in blobs:
        logs.append(json.loads(blob.download_as_text()))
    
    if not logs:
        return None

    current_df = pd.json_normalize(logs)
    current_df.columns = [c.split('.')[-1] for c in current_df.columns]
    
    # Path corrected for Docker environment
    ref_path = os.path.join(os.getcwd(), "data/processed/train_set.csv")
    if not os.path.exists(ref_path):
        return "Error: Reference data not found at " + ref_path

    reference_df = pd.read_csv(ref_path)
    
    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    
    output_html = "drift_report.html"
    report.save_html(output_html)
    return output_html