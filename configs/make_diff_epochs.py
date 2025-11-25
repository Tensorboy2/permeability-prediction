import yaml
import itertools
import os
file = os.path.dirname(__file__)

# Where to put YAMLs and Slurm scripts
YAML_DIR = os.path.join(file,"diffnum_epochs")
SLURM_DIR = YAML_DIR
os.makedirs(YAML_DIR, exist_ok=True)

# Parameter grid
models = [
          "ViT_T16",
          "ConvNeXtTiny", 
          "ConvNeXtSmall",
          "ViT_S16",
        "ResNet50",
        "ResNet101"
          ]
epochs = [200,300,400,500,1000]

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
        "lr": 0.0008,
        "batch_size": 128,
        # "num_epochs": 500,
        "warmup_steps": 1000,
        "weight_decay": 0.1,
        "decay": "cosine",
    }
}


slurm_script_paths = []
for model, epoch in itertools.product(models,epochs):
    name = f"{model}_{epoch}_epochs"
    yaml_path = os.path.join(YAML_DIR,f"{name}.yaml")
    # Build experiment dict
    exp = {
        "experiments": [
            {
                "model": model,
                "save_model_path": f"{name}.pth",
                "save_path": f"{name}.csv",
                "hyperparameters": {**common["hyperparameters"],"num_epochs": epoch,"clip_grad": False if model.startswith("ViT") else True},
                "data": common["data"],
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
launcher_path = os.path.join(file, "submit_all_jobs_diffnum_epochs.sh")
with open(launcher_path, "w") as f:
    f.write("#!/bin/bash\n")
    for script in slurm_script_paths:
        f.write(f"sbatch {script}\n")
print(f"Launcher script written: {launcher_path}")
