FROM python:3.12-slim AS base
WORKDIR /app

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc curl && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "dvc[gs]"

RUN dvc init --no-scm -f
RUN dvc remote add -d storage gs://dtu-mlops-group99-data

COPY data/raw.dvc data/raw.dvc
COPY data/processed.dvc data/processed.dvc

COPY pyproject.toml requirements.txt ./
RUN pip install -r requirements.txt --no-cache-dir

COPY src/ src/
COPY configs/ configs/
COPY tests/ tests/

RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["sh", "-c", "dvc pull -r storage && python -u src/project99/train.py"]
