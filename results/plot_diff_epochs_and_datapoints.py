'''
plot_diff_num_datapoints.py

Module for plotting the R^2 accuracy, the mean square error and the max R^2 for different dataset sizes.
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from matplotlib.lines import Line2D
import matplotlib as mpl
path = os.path.dirname(__file__)

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

sns.set_theme(style="whitegrid")


models = {
    "ConvNeXt-Small": { 
        "2000": "ConvNeXtSmall_2000_datapoints_run.csv",
        "4000": "ConvNeXtSmall_4000_datapoints_run.csv",
        "6000": "ConvNeXtSmall_6000_datapoints_run.csv",
        "8000": "ConvNeXtSmall_8000_datapoints_run.csv",
        "10000": "ConvNeXtSmall_10000_datapoints_run.csv",
        "12000": "ConvNeXtSmall_12000_datapoints_run.csv",
        "14000": "ConvNeXtSmall_14000_datapoints_run.csv",
        "16000": "ConvNeXtSmall_16000_datapoints_run.csv",
        "18000": "ConvNeXtSmall_18000_datapoints_run.csv",
        "20000": "convnextsmall_extralong.csv",
    },
    "ConvNeXt-Tiny": { 
        "2000": "ConvNeXtTiny_2000_datapoints_run.csv",
        "4000": "ConvNeXtTiny_4000_datapoints_run.csv",
        "6000": "ConvNeXtTiny_6000_datapoints_run.csv",
        "8000": "ConvNeXtTiny_8000_datapoints_run.csv",
        "10000": "ConvNeXtTiny_10000_datapoints_run.csv",
        "12000": "ConvNeXtTiny_12000_datapoints_run.csv",
        "14000": "ConvNeXtTiny_14000_datapoints_run.csv",
        "16000": "ConvNeXtTiny_16000_datapoints_run.csv",
        "18000": "ConvNeXtTiny_18000_datapoints_run.csv",
        "20000": "ConvNeXtTiny_500_epochs.csv",
    },
    "ViT-S16": {
        "2000": "ViT_S16_2000_datapoints_run.csv",
        "4000": "ViT_S16_4000_datapoints_run.csv",
        "6000": "ViT_S16_6000_datapoints_run.csv",
        "8000": "ViT_S16_8000_datapoints_run.csv",
        "10000": "ViT_S16_10000_datapoints_run.csv",
        "12000": "ViT_S16_12000_datapoints_run.csv",
        "14000": "ViT_S16_14000_datapoints_run.csv",
        "16000": "ViT_S16_16000_datapoints_run.csv",
        "18000": "ViT_S16_18000_datapoints_run.csv",
        "20000": "ViT_S16_500_epochs.csv",
        # "20000": "vit_s16_gradient_clip_long_test_3.csv",
    },
    "ViT-T16": {
        "2000": "ViT_T16_2000_datapoints_run.csv",
        "4000": "ViT_T16_4000_datapoints_run.csv",
        "6000": "ViT_T16_6000_datapoints_run.csv",
        "8000": "ViT_T16_8000_datapoints_run.csv",
        "10000": "ViT_T16_10000_datapoints_run.csv",
        "12000": "ViT_T16_12000_datapoints_run.csv",
        "14000": "ViT_T16_14000_datapoints_run.csv",
        "16000": "ViT_T16_16000_datapoints_run.csv",
        "18000": "ViT_T16_18000_datapoints_run.csv",
        "20000": "ViT_T16_500_epochs.csv",
    },
}


def plot_fit(models,mode='datapoints', save_path="diff_num_datapoints.pdf"):
    # Color maps per model:
    colormaps = {
        "ConvNeXt-Small": plt.cm.viridis,
        "ConvNeXt-Tiny": plt.cm.plasma,
        "ViT-S16": plt.cm.inferno,
        "ViT-T16": plt.cm.cividis,
        "ResNet-50": plt.cm.magma,
        "ResNet-101": plt.cm.coolwarm,
    }
    

    # Normalize based on number of datapoints:
    datapoint_keys = list(models["ConvNeXt-Small"].keys())
    datapoint_values = np.array([int(k) for k in datapoint_keys])
    norm = plt.Normalize(vmin=datapoint_values.min(), vmax=datapoint_values.max())

    # Store all model info:
    model_data = {}
    for model_name, files in models.items():
        model_data[model_name] = []
        for epochs, filename in files.items():
            df = pd.read_csv(os.path.join(path, filename))
            ep_int = int(epochs)
            color = colormaps[model_name](norm(ep_int))
            model_data[model_name].append({
                "epochs": ep_int,
                "df": df,
                "color": color,
                "filename": filename
            })

    # Plot max R^2 for each dataset size:
    data_records = []
    best_overall = {}
    for model_name, runs in model_data.items():
        best_overall[model_name] = {"r2": -np.inf, "mse":np.inf, "epoch": None, "file": None, "trained_epochs": None}
        for run in runs:
            df = run["df"]
            max_r2_idx = df["test_r2"].idxmax()
            max_r2 = df.loc[max_r2_idx, "test_r2"]
            max_epoch = df.loc[max_r2_idx, "epoch"] if "epoch" in df.columns else max_r2_idx
            max_mse = df.loc[max_r2_idx, "test_mse"]
            # Print summary for this CSV
            print(f"[{model_name}] CSV={run['filename']} | trained_epochs={run['epochs']} "
                  f"| max R²={max_r2:.4f} at epoch {max_epoch}")

            # Track best overall for this model
            if max_r2 > best_overall[model_name]["r2"]:
                best_overall[model_name] = {
                    "r2": max_r2,
                    "mse": max_mse,
                    "epoch": max_epoch,
                    "file": run["filename"],
                    "trained_epochs": run["epochs"]
                }

            data_records.append({
                "model": model_name,
                "num_datapoints": run["epochs"],   # x-axis
                "max_r2": 1 - max_r2,
                "color": run["color"],
            })

    print("\n=== Best results per model ===")
    for model_name, best in best_overall.items():
        print(f"[{model_name}] best R²={best['r2']:.6f}, MSE={best['mse']:.6f} "
              f"(epoch {best['epoch']} in {best['file']} with {best['trained_epochs']} trained epochs)")


    df_points = pd.DataFrame(data_records)
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    })
    plt.figure(figsize=(3.5, 3.5))
    # palette = {model: colormaps[model](0.8) for model in models.keys()}
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
    for model, color, marker in zip(df_points['model'].unique(), colors, markers):
        subset = df_points[df_points['model'] == model]
        plt.scatter(
            subset['num_datapoints'],
            subset['max_r2'],
            s=60,
            edgecolor=color,
            facecolor='none',  # hollow
            marker=marker,
            label=model,
            linewidth=1.0
        )
    plt.ylabel(r"$1-R^2$")
    plt.yscale('log')
    if mode =='datapoints':
        plt.xlabel("Training Samples(x1000)")
        # plt.xscale('log')
        plt.yticks([0.1, 0.05,0.04,0.03,0.02, 0.01, 0.005, 0.004, 0.003, 0.002, 0.001], 
                ['0.1', '0.05', '0.04', '0.03', '0.02', '0.01', '0.005', '0.004', '0.003', '0.002', '0.001'])
        plt.xticks([2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000],
                # ['2k', '4k', '6k', '8k', '10k', '9.6k', '11.2k', '12.2k', '14.4k', '16k']
                [f"{(x*0.8/1000):.1f}" for x in [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]]
                )
    if mode =='epochs':
        plt.xlabel("Number of Training Epochs")
        plt.xscale('log')
        plt.xticks([100, 200, 300, 400, 500, 600, 1000], [100, 200, 300, 400, 500, 600, 1000])
        plt.yticks([0.1, 0.05,0.04,0.03,0.02, 0.01, 0.005, 0.004, 0.003, 0.002, 0.001], 
                ['0.1', '0.05', '0.04', '0.03', '0.02', '0.01', '0.005', '0.004', '0.003', '0.002', '0.001'])
    plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(path, save_path), bbox_inches='tight')

plot_fit(models,mode='datapoints')

models = {
    "ConvNeXt-Small": {
        "1000": "ConvNeXtSmall_all.csv",
        "500": "ConvNeXtSmall_500_epochs.csv",
        "400": "ConvNeXtSmall_400_epochs.csv",
        "300": "ConvNeXtSmall_300_epochs.csv",
        "200": "ConvNeXtSmall_200_epochs.csv",
        "100": "ConvNeXtSmall_100_epochs.csv",
        },
    "ConvNeXt-Tiny": {
        "1000": "ConvNeXtTiny_all.csv",
        "600": "ConvNeXtTiny_600_epochs.csv",
        "500": "ConvNeXtTiny_500_epochs.csv",
        "400": "ConvNeXtTiny_400_epochs.csv",
        "300": "ConvNeXtTiny_300_epochs.csv",
        "200": "ConvNeXtTiny_200_epochs.csv",
        "100": "ConvNeXtTiny_100_epochs.csv",
        },
    "ViT-S16": {
        "1000": "vit_s16_gradient_clip_long_test_3.csv",
        "600": "ViT_S16_600_epochs.csv",
        "500": "ViT_S16_500_epochs.csv",
        "400": "ViT_S16_400_epochs.csv",
        "300": "ViT_S16_300_epochs.csv",
        "200": "ViT_S16_200_epochs.csv",
        "100": "ViT_S16_100_epochs.csv",
        },
    "ViT-T16": {
        "1000": "ViT_T16_all.csv",
        "600": "ViT_T16_600_epochs.csv",
        "500": "ViT_T16_500_epochs.csv",
        "400": "ViT_T16_400_epochs.csv",
        "300": "ViT_T16_300_epochs.csv",
        "200": "ViT_T16_200_epochs.csv",
        "100": "ViT_T16_100_epochs.csv",
        },
    "ResNet-50": {
        "1000": "ResNet50_all.csv",
        # "500": "resnet50_gradient_clip_500_epochs.csv",
        "500": "ResNet50_500_epochs.csv",
        "400": "ResNet50_400_epochs.csv",
        "300": "ResNet50_300_epochs.csv",
        "200": "ResNet50_200_epochs.csv",
        "100": "resnet50_gradient_clip_test.csv",
        },
    "ResNet-101": {
        "1000": "ResNet101_all.csv",
        "500": "ResNet101_500_epochs.csv",
        "400": "ResNet101_400_epochs.csv",
        "300": "ResNet101_300_epochs.csv",
        "200": "ResNet101_200_epochs.csv",
        "100": "resnet101_metrics_all_models_2.csv",
        },
}

plot_fit(models,mode='epochs', save_path="diff_num_epochs.pdf")
