import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from project99.monitoring import generate_drift_report

app = FastAPI()

@app.get("/report", response_class=HTMLResponse)
async def get_report():
    report_file = generate_drift_report()
    
    if report_file and os.path.exists(report_file):
        with open(report_file, "r") as f:
            return f.read()
    
    return "<h1>No data collected yet or Reference data missing.</h1>"

@app.get("/health")
async def health():
    return {"status": "monitoring api is up"}