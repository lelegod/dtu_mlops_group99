FROM python:3.10-slim
WORKDIR /app
COPY models/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ["data/processed/train_set.csv", "./data/processed/"]
COPY configs/config.yaml .
COPY models/train.py.py ./train.py
CMD ["python", "train.py"]