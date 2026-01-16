FROM python:3.12-slim AS base

WORKDIR /app

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY configs/ configs/

# Copy data (later with gcp)
COPY data/processed/train_set.csv data/processed/train_set.csv
COPY data/processed/test_set.csv data/processed/test_set.csv

RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

ENTRYPOINT ["python", "-u", "src/project99/train.py"]
