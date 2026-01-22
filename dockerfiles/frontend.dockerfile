FROM python:3.12-slim
WORKDIR /app

# Install system dependencies for curl and build tools
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc curl && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Install requirements first for better caching
COPY requirements_frontend.txt .
RUN pip install --no-cache-dir -r requirements_frontend.txt

# Copy source and install project
COPY . . 
RUN pip install --no-deps .

EXPOSE 8501

# FIXED: Streamlit health check path changed to /healthz or /_stcore/health
# Cloud Run also prefers no HEALTHCHECK in the Dockerfile if you configure it in YAML, 
# but this is the correct command if you keep it:
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Environment variables for Cloud Run compatibility
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Entrypoint using the shell form to ensure ENV variables are respected
ENTRYPOINT ["streamlit", "run", "src/project99/frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]