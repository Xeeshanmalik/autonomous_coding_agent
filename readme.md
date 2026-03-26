Running LLM agent

docker network create research-net

docker run -d \
  --name local-deepseek-backend \
  --network research-net \
  -p 8085:8080 \
  inference_server_autonomous_agent:latest \
  -t 8

docker run -it -d \
  --name ai-researcher \
  --network research-net \
  -e LLM_BASE_URL="http://local-deepseek-backend:8080/v1" \
  -e LLM_MODEL="deepSeek-R1-Distill-Qwen-32B" \
  ai-researcher:latest   

 docker run -d \
  --name ai-frontend-ui \
  --network research-net \
  -p 3000:8080 \
  ai-frontend-ui:latest 