FROM python:3.12-slim
WORKDIR /app

RUN apt update && apt install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/

ENV PYTHONPATH=/app/src

EXPOSE 8080

CMD ["uvicorn", "project99.monitoring_api:app", "--host", "0.0.0.0", "--port", "8080"]