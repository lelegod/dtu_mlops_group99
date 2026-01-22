FROM python:3.12-slim

WORKDIR /app

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc curl && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements_frontend.txt .
RUN pip install --no-cache-dir -r requirements_frontend.txt

COPY src/project99/frontend.py .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:${PORT:-8501}/_stcore/health || exit 1

ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

CMD streamlit run frontend.py --server.port=${PORT:-8501} --server.address=0.0.0.0
