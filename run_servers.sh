
cd /home/da02/code/RouteLLM/fake_server/weak/
nohup uvicorn main:app --host 0.0.0.0 --port 8000 &> weak-server.log < /dev/null &


cd /home/da02/code/RouteLLM/fake_server/strong/
nohup uvicorn main:app --host 0.0.0.0 --port 8001 &> strong-server.log < /dev/null &

cd /home/da02/code/RouteLLM/
python3 -m routellm.openai_server --strong-model http://localhost:8001/v1 --weak-model http://localhost:8000/v1  --routers mf --config ./routes.yaml