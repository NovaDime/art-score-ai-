#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$PROJECT_DIR/.app.pid"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/server.log"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
APP_MODULE="${APP_MODULE:-main:app}"
CONDA_ENV="${CONDA_ENV:-artscore}"

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

if [ -f "$PID_FILE" ]; then
  OLD_PID=$(cat "$PID_FILE" 2>/dev/null || true)
  if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "应用已经在运行中，PID=$OLD_PID"
    echo "如需重启，请先执行 ./stop.sh"
    exit 1
  else
    rm -f "$PID_FILE"
  fi
fi

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

echo "启动 8.0 Beta 服务器模式..."
echo "项目目录: $PROJECT_DIR"
echo "监听地址: http://$HOST:$PORT"
echo "日志文件: $LOG_FILE"

nohup python -m uvicorn "$APP_MODULE" --host "$HOST" --port "$PORT" > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

sleep 3

if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "服务器启动成功，PID=$(cat "$PID_FILE")"
else
  echo "启动失败，请查看日志: $LOG_FILE"
  exit 1
fi
