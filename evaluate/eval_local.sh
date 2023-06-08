#!/bin/bash
identifier=$(date +%s)_eval
#domains=(MountainCar MarsRover HVAC PowerGen_continuous UAV_continuous Reservoir_continuous RecSim RaceCar)
domains=(RecSim)
insts=(1 3 5)


for env_name in "${domains[@]}"; do
    log_path=/home/roni-lab/Research/RDDL-demo-agent-IPPC-23/logs/${env_name}/${identifier}
    mkdir -p ${log_path}
    for inst in "${insts[@]}"; do
        echo "Evaluating ${env_name}-${inst}"
        log_file=${log_path}/${inst}
        python /home/roni-lab/Research/RDDL-demo-agent-IPPC-23/main.py ${env_name} ${inst}c 1 > ${log_file}.out 2> ${log_file}.err
    done
done

# baselines=(noop random plan)
# for baseline in "${baselines[@]}"; do
#     for inst in "${insts[@]}"; do
#         slurm_log_file=${inst}_${baseline}
#         jid=$(sbatch --export=ALL,base_run_name=${identifier},env=${env_name},alg=${baseline},inst=${inst} --parsable -J e-${env_name}-${inst}-${baseline} -o ${SLURM_LOG_PATH}/${slurm_log_file}.out  -e ${SLURM_LOG_PATH}/${slurm_log_file}.err /N/u/palchatt/BigRed3/IPC23/evaluate/eval_baseline.sh)
#         echo "Env: ${env_name}. Inst: ${inst}. Mode: ${baseline}, JID: ${jid}"
#         jids+=($jid)
#     done
# done
