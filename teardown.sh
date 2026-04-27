#!/bin/bash
# Runs ON the VM after tarball is uploaded. Called by redeploy-swarm.sh.
set -e
pkill -f autoresearch_swarm.py 2>/dev/null || true
pkill -f 'python.*train.py' 2>/dev/null || true
sleep 1
sudo rm -rf /tmp/astar
mkdir -p /tmp/astar
cd /tmp/astar
tar xzf /tmp/astar-deploy.tar.gz
pip3 install --break-system-packages -q numpy scipy xgboost scikit-learn pydantic httpx 2>&1 | tail -1
python3 -c 'import numpy,scipy,xgboost,sklearn,pydantic,httpx;print("imports OK")'
VM_ID=$(hostname | sed 's/ainm-astar-//')
export GOOGLE_API_KEY='AIzaSyDneTtqxEnKB3ZZeQ9MpPKeAxLyoWvbYQM'
export MODEL_ID='gemini-3.1-flash-lite-preview'
export PARALLEL_XGB='18'
export VM_ID
export VM_FOCUS='general'
export PYTHONUNBUFFERED=1
nohup python3 -u autoresearch_swarm.py > swarm.log 2>&1 &
sleep 3
if pgrep -f autoresearch_swarm.py > /dev/null; then
    echo "STARTED on $(hostname)"
    tail -5 swarm.log 2>/dev/null
else
    echo "FAILED on $(hostname)"
    tail -15 swarm.log 2>/dev/null
    exit 1
fi
