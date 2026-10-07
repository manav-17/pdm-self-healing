FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements_api.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY api/ api/
COPY ml/ ml/
COPY agents/ agents/
COPY edge/ edge/
COPY data/processed/ data/processed/
ENV GCP_PROJECT=project-c3e3f3ae-71cf-45cb-800 BQ_DATASET=pdm USE_LLM=false PORT=8080
EXPOSE 8080
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
