#!/usr/bin/env bash
set +e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$PROJECT_DIR/.app.pid"

cd "$PROJECT_DIR"

if [ -f "$PID_FILE" ]; then
  PID=$(cat "$PID_FILE" 2>/dev/null)
  if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
    echo "正在停止应用，PID=$PID"
    kill "$PID"
    sleep 2
    if kill -0 "$PID" 2>/dev/null; then
      echo "普通停止未成功，执行强制终止..."
      kill -9 "$PID"
    fi
    rm -f "$PID_FILE"
    echo "应用已停止"
    exit 0
  else
    echo "PID 文件存在，但进程已不在运行，清理 PID 文件"
    rm -f "$PID_FILE"
  fi
fi

echo "未发现 PID 文件，尝试按 uvicorn 关键字兜底清理..."
pkill -f "uvicorn main:app" 2>/dev/null

echo "停止命令执行完毕"
