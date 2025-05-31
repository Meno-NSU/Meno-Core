cat backend-process.txt | xargs kill -TERM
sleep 3
nohup uvicorn backend_api:app --host 127.0.0.1 --port 8888 > backend.log 2>&1 &
echo $! > backend-process.txt