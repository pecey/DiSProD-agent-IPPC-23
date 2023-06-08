#!/bin/bash
# modes=(complete no_var sampling)
identifier=$(date +%s)_eval
insts=(1 3 5)

env_name="HVAC"
jids=()

echo "Evaluating ${env_name}"
SLURM_LOG_PATH=/N/u/palchatt/BigRed3/RDDL-demo-agent-IPPC-23/slurm_logs/${env_name}/${identifier}
mkdir -p ${SLURM_LOG_PATH}

#for mode in "${modes[@]}"; do
for inst in "${insts[@]}"; do
    slurm_log_file=${inst}
    jid=$(sbatch --export=ALL,base_run_name=${identifier},env=${env_name},inst=${inst} --parsable -J e-${env_name}-${inst} -o ${SLURM_LOG_PATH}/${slurm_log_file}.out  -e ${SLURM_LOG_PATH}/${slurm_log_file}.err /N/u/palchatt/BigRed3/RDDL-demo-agent-IPPC-23/evaluate/eval.sh)
    echo "Env: ${env_name}. Inst: ${inst}. JID: ${jid}"
    jids+=($jid)
done
#done

# baselines=(noop random plan)
# for baseline in "${baselines[@]}"; do
#     for inst in "${insts[@]}"; do
#         slurm_log_file=${inst}_${baseline}
#         jid=$(sbatch --export=ALL,base_run_name=${identifier},env=${env_name},alg=${baseline},inst=${inst} --parsable -J e-${env_name}-${inst}-${baseline} -o ${SLURM_LOG_PATH}/${slurm_log_file}.out  -e ${SLURM_LOG_PATH}/${slurm_log_file}.err /N/u/palchatt/BigRed3/IPC23/evaluate/eval_baseline.sh)
#         echo "Env: ${env_name}. Inst: ${inst}. Mode: ${baseline}, JID: ${jid}"
#         jids+=($jid)
#     done
# done
