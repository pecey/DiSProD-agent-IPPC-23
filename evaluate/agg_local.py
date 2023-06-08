import os
import tabulate
import sys

domains = ["MarsRover", "RecSim", "RaceCar", "MountainCar", "HVAC", "UAV_continuous", "PowerGen_continuous", "Reservoir_continuous"]
slurm_log_path = "/home/roni-lab/Research/RDDL-demo-agent-IPPC-23/logs"

def main():
    
    # out_files = [f for f in os.listdir(f"{path}/{result_dir}") if ".out" in f]
    op_table = []
    for env in domains:
        path = f"{slurm_log_path}/{env}"
        eval_dirs = [d for d in os.listdir(path) if "_eval" in d]
        result_dir = sorted(eval_dirs)[-1]
        op_entry = {"domain": env}
        for inst in [1,3,5]:
            out_file_path = f"{path}/{result_dir}/{inst}.out"
            if os.path.exists(out_file_path):
                with open(out_file_path) as f:
                    try:
                        data = f.readlines()[-1]
                        if "Mean" not in data:
                            op_entry[inst] = "time out"
                        else:
                            stats = [e.strip().split(":")[1] for e in data.split(",")[:2]]
                            mean = float(stats[0].strip())
                            sd = float(stats[1].strip())
                            op_entry[inst] = f"{mean:.2f} + {sd:.2f}"
                    except IndexError:
                        op_entry[inst] = "time out"
        op_table.append(op_entry)
    return op_table


if __name__ == "__main__":
    scores = main()
    print(tabulate.tabulate(scores, headers="keys"))