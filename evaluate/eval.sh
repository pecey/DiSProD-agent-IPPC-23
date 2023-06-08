#!/bin/bash

#SBATCH -p general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=0-2:00:00
#SBATCH --mem=32GB
#SBATCH -A general

env_name=$env
inst=$inst

restart=$restart
depth=$depth
n_episodes=10

today=$(date +'%m-%d-%Y')

module load python/3.9.8

start_time=`(date +%s)`
echo Start time: ${start_time}

source ${HOME}/IPC23-venv/bin/activate
cd /N/u/palchatt/BigRed3/RDDL-demo-agent-IPPC-23
# export IPC_PATH=/N/u/palchatt/BigRed3/RDDL-demo-agent-IPPC-23

# To evaluate model
PYTHONPATH=. python main.py ${env_name} ${inst}c ${n_episodes}

end_time=`(date +%s)`
echo End time: ${end_time}
