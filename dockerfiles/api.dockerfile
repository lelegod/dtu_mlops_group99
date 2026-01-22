FROM python:3.12-slim AS base
WORKDIR /app

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc curl && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY pyproject.toml .
COPY README.md .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY src/ src/
COPY configs/ configs/

# Install the project in editable mode so project99 is a recognized module
RUN pip install --no-cache-dir -e .

EXPOSE 8000

# Ensure we point to the correct file name (api.py -> project99.api)
# We use the $PORT variable provided by Cloud Run, defaulting to 8000
ENTRYPOINT ["sh", "-c", "uvicorn project99.api:app --host 0.0.0.0 --port ${PORT:-8000}"]