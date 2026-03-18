#!/bin/bash
# Run both FastAPI backend and React frontend

DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Starting Insurance Claim Search..."
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo ""

# Start backend
cd "$DIR"
"$DIR/.venv_bge/bin/python3.12" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Start frontend
cd "$DIR/frontend"
npm run dev &
FRONTEND_PID=$!

echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
