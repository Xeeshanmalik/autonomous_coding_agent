Running LLM agent

docker network create research-net

docker run -d \
  --name local-qwen-backend \
  --network research-net \
  -p 8082:8080 \
  inference_server_autonomous_agent:latest \
  -t 8

docker run -it \
  --name ai-researcher \
  --network research-net \
  -e LLM_BASE_URL="http://local-qwen-backend:8080/v1" \
  -e LLM_MODEL="qwen2.5-coder-7b" \
  ai-researcher:latest   

 docker run -d \
  --name ai-frontend-ui \
  --network research-net \
  -p 3000:8080 \
  autoresearch-frontend:latest 