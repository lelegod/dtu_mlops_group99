FROM python:3.10-slim
WORKDIR /app/src/project99
COPY models/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ["data/processed/train_set.csv", "./data/processed/"]
COPY configs/config.yaml .
COPY src/ ./src/
COPY data/ /app/data/
COPY src/project99/train.py ./train.py
ENV PYTHONPATH="${PYTHONPATH}:/app/src"
CMD ["python", "train.py"]




