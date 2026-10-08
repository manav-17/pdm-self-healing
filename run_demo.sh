#!/bin/zsh
cd ~/Desktop/LTM_TP && source venv/bin/activate
pkill -f edge/pipeline.py 2>/dev/null
open -a Docker
until docker info >/dev/null 2>&1; do sleep 3; done
docker start mosquitto >/dev/null && echo "MQTT broker running"

export API_URL=https://pdm-api-363641793280.asia-south1.run.app
export API_TOKEN=$(gcloud auth print-identity-token)
export USE_LLM=${USE_LLM:-true}
export GEMINI_MODEL=${GEMINI_MODEL:-gemini-flash-lite-latest}

python3 -u edge/pipeline.py > /tmp/pipeline.log 2>&1 &
PIPE_PID=$!
sleep 8
python3 edge/simulator.py --engines 1 --rate 0.2
echo "processing..."
for i in {1..180}; do grep -q "c192 |" /tmp/pipeline.log && break; sleep 10; done
sleep 3; kill $PIPE_PID 2>/dev/null
echo "escalated: $(grep -c 'queued' /tmp/pipeline.log)   llm: $(grep -c 'queued \[llm\]' /tmp/pipeline.log)   template: $(grep -c 'queued \[template\]' /tmp/pipeline.log)   API errors: $(grep -c 'API error' /tmp/pipeline.log)"
echo "Dashboard: https://pdm-dashboard-necm52e4nq-el.a.run.app"
