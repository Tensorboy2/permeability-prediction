import yaml
import itertools
import os
import argparse

file = os.path.dirname(__file__)

# -----------------------------
# Parse mode
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["herbie", "slurm"], default="slurm", help="Mode for experiment generation")
parser.add_argument("--models", choices=["both","convnext", "vit"], default="both", help="Mode for experiment generation")
args = parser.parse_args()
mode = args.mode

# Where to put YAMLs and Slurm scripts
exp_name = "datapoints_run"
YAML_DIR = os.path.join(file, exp_name)
SLURM_DIR = YAML_DIR
os.makedirs(YAML_DIR, exist_ok=True)

# Parameter grid
if args.models == "both":
    models = ["ViT_T16",
              "ConvNeXtTiny"]
elif args.models == "convnext":
    models = ["ConvNeXtTiny"]
elif args.models == "vit":
        models = ["ViT_T16"]

data_set_sizes = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000]

# optional fixed fields
common = {
    "data": {
        "hflip": False,
        "vflip": False,
        "rotate": False,
        "group": True,
        "test_size": 0.2
    },
    "hyperparameters": {
        "lr": 0.0008,
        "batch_size": 128,
        "warmup_steps": 1000,
        "weight_decay": 0.1,
        "decay": "cosine"
    }
}

slurm_script_paths = []

if mode == "herbie":
    # Combine all experiments into a single YAML
    all_experiments = []
    for model, data_set_size in itertools.product(models, data_set_sizes):
        name = f"{model}_{data_set_size}_{exp_name}"
        all_experiments.append({
            "model": model,
            "save_model_path": f"{name}.pth",
            "save_path": f"{name}.csv",
            "hyperparameters": {**common["hyperparameters"], 
                                "clip_grad": False if model.startswith("ViT") else True,
                                "num_epochs": 500 if model.startswith("ViT") else 500},
            "data": {**common["data"], "num_samples": data_set_size}
        })
    yaml_path = os.path.join(YAML_DIR, f"all_experiments.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump({"experiments": all_experiments}, f)
    print(f"Generated normal mode YAML with {len(all_experiments)} experiments: {yaml_path}")

else:  # slurm mode
    for model, data_set_size in itertools.product(models, data_set_sizes):
        name = f"{model}_{data_set_size}_{exp_name}"
        yaml_path = os.path.join(YAML_DIR, f"{name}.yaml")
        # Build experiment dict
        exp = {
            "experiments": [
                {
                    "model": model,
                    "save_model_path": f"{name}.pth",
                    "save_path": f"{name}.csv",
                    "hyperparameters": {**common["hyperparameters"], 
                                        "clip_grad": False if model.startswith("ViT") else True, 
                                        "num_epochs": 500 if model.startswith("ViT") else 500},
                    "data": {**common["data"], "num_samples": data_set_size}
                }
            ]
        }
        # Write YAML
        with open(yaml_path, "w") as f:
            yaml.dump(exp, f)
        # Write Slurm script
        slurm_script = f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

python main.py \"{yaml_path}\"
"""
        script_path = os.path.join(SLURM_DIR, f"{name}.sh")
        with open(script_path, "w") as f:
            f.write(slurm_script)
        slurm_script_paths.append(script_path)
        print(f"Generated: {name}")

    # Write a .sh script to launch all slurm jobs
    launcher_path = os.path.join(file, f"submit_all_jobs_{exp_name}.sh")
    with open(launcher_path, "w") as f:
        f.write("#!/bin/bash\n")
        for script in slurm_script_paths:
            f.write(f"sbatch {script}\n")
    print(f"Launcher script written: {launcher_path}")
