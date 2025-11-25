import yaml
import argparse
import os
file = os.path.dirname(__file__)

# -----------------------------
# Parse mode
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["herbie", "slurm"], default="slurm", help="Mode for experiment generation")
# parser.add_argument("--models", choices=["both","convnext", "vit"], default="both", help="Mode for experiment generation")
args = parser.parse_args()
mode = args.mode

# Where to put YAMLs and Slurm scripts
exp_name = "all_models_run_4"
YAML_DIR = os.path.join(file,exp_name)
SLURM_DIR = YAML_DIR
os.makedirs(YAML_DIR, exist_ok=True)

# Parameter grid
models = ["ViT_T16",
          "ViT_S16", # will already have been run
          "ConNextSmall", # will already have been run
          "ConvNeXtTiny",
            "ResNet50",
            "ResNet101"
          ]

# optional fixed fields
common = {
    "data": {
        "hflip": False,
        "vflip": False,
        "rotate": False,
        "group": True,
        "test_size": 0.2,
        "num_samples": None,
    },
    "hyperparameters": {
        "lr": 0.0001,
        "batch_size": 256,
        "num_epochs": 1000,
        "warmup_steps": 1000,
        "weight_decay": 0.1,
        "decay": "cosine"
    }
}


slurm_script_paths = []

if mode == "herbie":
    # Combine all experiments into a single YAML
    all_experiments = []
    for model in models:
        name = f"{model}_{exp_name}"
        all_experiments.append({
            "model": model,
            "save_model_path": f"{name}.pth",
            "save_path": f"{name}.csv",
            "hyperparameters": {**common["hyperparameters"], "clip_grad": False if model.startswith("ViT") else True},
            "data": common["data"],
        })
    # Write YAML
    yaml_path = os.path.join(YAML_DIR,f"all_models_run.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump({"experiments": all_experiments}, f)
    print(f"Generated normal mode YAML with {len(all_experiments)} experiments: {yaml_path}")

else:  # slurm mode
    for model in models:
        name = f"{model}_{exp_name}"
        yaml_path = os.path.join(YAML_DIR,f"{name}.yaml")
        # Build experiment dict
        exp = {
            "experiments": [
                {
                    "model": model,
                    "save_model_path": f"{name}.pth",
                    "save_path": f"{name}.csv",
                    "hyperparameters": {**common["hyperparameters"], "clip_grad": False if model.startswith("ViT") else True},
                    "data": common["data"],
                }
            ]
        }
        # Write YAML
        with open(yaml_path, "w") as f:
            yaml.dump(exp, f)
        print(f"Generated slurm mode YAML: {yaml_path}")
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