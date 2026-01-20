FROM python:3.12-slim AS base
WORKDIR /app

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc curl && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY README.md README.md

RUN pip install -r requirements.txt --no-cache-dir

COPY src/ src/
COPY configs/ configs/

RUN pip install . --no-deps --no-cache-dir

EXPOSE 8000

ENTRYPOINT ["uvicorn", "project99.api:app", "--host", "0.0.0.0", "--port", "8000"]