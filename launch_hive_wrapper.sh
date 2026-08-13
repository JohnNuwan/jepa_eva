#!/bin/bash
cd /home/aza/projects/jepa_eva
for g in 0 1 2; do
  nohup venv/bin/python3 launch_hive_v2.py $g 3 > logs_massive/worker_hive_${g}.log 2>&1 &
  sleep 3
done
echo '3 workers hive v2 launched'
