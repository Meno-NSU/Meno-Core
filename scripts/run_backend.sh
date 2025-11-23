cat backend-process.txt | xargs kill -TERM
sleep 3
nohup uv run uvicorn meno_core.api.main:app --host 127.0.0.1 --port 8888 > backend.log 2>&1 &
echo $! > backend-process.txt