#!/bin/zsh
cd ~/Desktop/LTM_TP && source venv/bin/activate
DATASET=${DATASET:-FD004}
ENGINE=${ENGINE:-1}
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
for i in {1..45}; do grep -q "subscribing to" /tmp/pipeline.log && break; kill -0 $PIPE_PID 2>/dev/null || break; sleep 2; done
if ! grep -q "subscribing to" /tmp/pipeline.log; then
  echo "pipeline did not start:"; tail -20 /tmp/pipeline.log; kill $PIPE_PID 2>/dev/null; exit 1
fi
head -1 /tmp/pipeline.log

python3 edge/simulator.py --dataset $DATASET --start-engine $ENGINE --engines 1 --rate 0.2 | tee /tmp/simulator.log
LAST=$(grep -o 'cycle=[0-9]*' /tmp/simulator.log | tail -1 | cut -d= -f2)
echo "processing (waiting for cycle $LAST)..."
for i in {1..240}; do grep -q "c${LAST} |" /tmp/pipeline.log && break; sleep 10; done
sleep 3; kill $PIPE_PID 2>/dev/null
echo "processed: $(grep -c ' | cond=' /tmp/pipeline.log)   rejected: $(grep -c 'REJECTED' /tmp/pipeline.log)   errors: $(grep -c '^ERROR' /tmp/pipeline.log)"
echo "escalated: $(grep -c 'queued' /tmp/pipeline.log)   llm: $(grep -c 'queued \[llm\]' /tmp/pipeline.log)   template: $(grep -c 'queued \[template\]' /tmp/pipeline.log)   API errors: $(grep -c 'API error' /tmp/pipeline.log)"
echo "conditions seen: $(grep -o 'cond=[0-9]' /tmp/pipeline.log | sort | uniq -c | tr '\n' ' ')"
echo "Dashboard: https://pdm-dashboard-necm52e4nq-el.a.run.app"
