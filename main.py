import argparse
import os
import subprocess
from datetime import datetime

def run_attack(script, dataset, seed, args, extra_args):
    log_file = os.path.join(args.output_dir, "log", dataset, f"{script}_seed_{seed}.log")
    command = (
        f"python -c \"from attacks.{script} import attack0; "
        f"attack0('{dataset}', {seed}, 'cuda:{args.cuda}' if '{args.cuda}' != 'None' else 'cpu', "
        f"{args.attack_node_arg}, file_path='{args.output_dir}', lr={args.lr}, tgt_lr={args.tgt_lr}, "
        f"eval_epoch={args.eval_epoch}, tgt_epoch={args.tgt_epoch}, "
        f"dropout={args.dropout}, model_performance={args.model_performance} {extra_args})\""
    )
    with open(log_file, "w") as log:
        subprocess.Popen(command, shell=True, stdout=log, stderr=log)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["amazoncomputer", "coauthorCS", "coauthorphysics", "amazonphoto", "dblp", "cora_full"], required=True)
    parser.add_argument("--cuda", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tgt_lr", type=float, default=1e-3)
    parser.add_argument("--eval_epoch", type=int, default=1000)
    parser.add_argument("--tgt_epoch", type=int, default=1000)
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--radium", type=float, default=0.005)
    parser.add_argument("--dropout", type=bool, default=False)
    parser.add_argument("--model_performance", type=bool, default=True)
    parser.add_argument("--warmup_epoch", type=int, default=400)
    parser.add_argument("--method", choices=["grain_nnd", "grain_ball", "age", "random", "cega"], required=True)
    
    args = parser.parse_args()
    
    args.attack_node_arg = 0.1 if args.dataset in ["amazoncomputer", "coauthorCS", "coauthorphysics", "amazonphoto", "dblp"] else 0.25
    
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.output_dir = os.path.join("./output", current_datetime)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "log"), exist_ok=True)
    
    with open(os.path.join(args.output_dir, "settings.txt"), "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}={value}\n")
    
    method_map = {
        "grain_nnd": "attack_0_grain_nnd",
        "grain_ball": "attack_0_grain_ball",
        "age": "attack_0_age",
        "random": "attack_0_random",
        "cega": "attack_0_cega"
    }
    
    attack_script = method_map[args.method]
    
    dataset_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "log", args.dataset), exist_ok=True)

    for seed in range(1, args.num_runs + 1):
        extra_args = ""
        if args.method == "grain_ball":
            extra_args = f", radium={args.radium}"
        elif args.method == "age":
            extra_args = f", warmup_epoch={args.warmup_epoch}"
        run_attack(attack_script, args.dataset, seed, args, extra_args)
        
if __name__ == "__main__":
    main()