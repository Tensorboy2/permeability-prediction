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
parser.add_argument("--models", choices=["both","convnext", "vit"], default="both", help="Which models to use")
args = parser.parse_args()
mode = args.mode

# Where to put YAMLs and Slurm scripts
exp_name = "dataaug_run"
YAML_DIR = os.path.join(file, exp_name)
SLURM_DIR = YAML_DIR
os.makedirs(YAML_DIR, exist_ok=True)

# Parameter grid
if args.models == "both":
    models = ["ViT_S16", "ConvNeXtSmall"]
elif args.models == "convnext":
    models = ["ConvNeXtSmall"]
elif args.models == "vit":
    models = ["ViT_S16"]

augmentation_variants = [
    {"hflip": False, "vflip": False, "rotate": False, "group": False},
    {"hflip": True, "vflip": False, "rotate": False, "group": False},
    {"hflip": False, "vflip": True, "rotate": False, "group": False},
    {"hflip": False, "vflip": False, "rotate": True, "group": False},
    {"hflip": True, "vflip": True, "rotate": False, "group": False},
    {"hflip": True, "vflip": True, "rotate": True, "group": False},
    {"hflip": False, "vflip": False, "rotate": False, "group": True},
]

# optional fixed fields
common = {
    "data": {
        "val_size": 0.2,
        "num_samples": None
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
    for model, augmentation_variant in itertools.product(models, augmentation_variants):
        active = [k for k, v in augmentation_variant.items() if v]
        active_str = "_".join(active) if active else "none"
        name = f"{model}_{active_str}_{exp_name}"
        all_experiments.append({
            "model": model,
            "save_model_path": f"{name}.pth",
            "save_path": f"{name}.csv",
            "hyperparameters": {**common["hyperparameters"], 
                                "clip_grad": False if model.startswith("ViT") else True,
                                "num_epochs": 500 if model.startswith("ViT") else 500},
            "data": {**common["data"], **augmentation_variant}
        })
    yaml_path = os.path.join(YAML_DIR, f"all_experiments.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump({"experiments": all_experiments}, f)
    print(f"Generated normal mode YAML with {len(all_experiments)} experiments: {yaml_path}")

else:  # slurm mode
    for model, augmentation_variant in itertools.product(models, augmentation_variants):
        active = [k for k, v in augmentation_variant.items() if v]
        active_str = "_".join(active) if active else "none"
        name = f"{model}_{active_str}_{exp_name}"
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
                    "data": {**common["data"], **augmentation_variant}
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
