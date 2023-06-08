import os
import tabulate
import sys

planners = ["no_var", "sampling", "complete", "noop", "random", "plan"]
slurm_log_path = os.getenv("IPC_SLURM_LOG_PATH")

def main(path):
    eval_dirs = [d for d in os.listdir(path) if "_eval" in d]
    result_dir = sorted(eval_dirs)[-1]
    # out_files = [f for f in os.listdir(f"{path}/{result_dir}") if ".out" in f]
    op_table = []
    for planner in planners:
        op_entry = {"planner": planner}
        for inst in range(0, 6):
            out_file_path = f"{path}/{result_dir}/{inst}_{planner}.out"
            if os.path.exists(out_file_path):
                with open(out_file_path) as f:
                    try:
                        data = f.readlines()[-2]
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
    env = sys.argv[1]
    path = f"{slurm_log_path}/{env}"
    scores = main(path)
    print(tabulate.tabulate(scores, headers="keys"))