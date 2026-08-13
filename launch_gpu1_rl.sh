#!/bin/bash
cd /home/aza/projects/jepa_eva
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.12

# Worker 0: PPO
nohup venv/bin/python3 massive_launch_rl.py --algo ppo --groupe 0 --total 3 > logs_massive/ppo_hive.log 2>&1 &
echo "PPO launched"
sleep 5

# Worker 1: DQN
nohup venv/bin/python3 massive_launch_rl.py --algo dqn --groupe 1 --total 3 > logs_massive/dqn_hive.log 2>&1 &
echo "DQN launched"
sleep 5

# Worker 2: TD3
nohup venv/bin/python3 massive_launch_rl.py --algo td3 --groupe 2 --total 3 > logs_massive/td3_hive.log 2>&1 &
echo "TD3 launched"

echo "ALL_LAUNCHED"
