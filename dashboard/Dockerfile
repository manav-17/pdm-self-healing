FROM python:3.11-slim
WORKDIR /app
COPY dashboard/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY dashboard/ dashboard/
COPY agents/risk_agent.py agents/risk_agent.py
ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "streamlit run dashboard/app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
