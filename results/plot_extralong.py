'''
plot_dataaug.py

Module for plotting the R^2 accuracy and the mean square error form training with different data augmentation techniques.
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




def load_models_info(augmentation_variants,cmap="Reds"):
    colors = sns.color_palette(cmap, n_colors=len(augmentation_variants))#[::-1]
    models_info = {}
    for (label, file), color in zip(augmentation_variants.items(), colors):
        try:
            df = pd.read_csv(os.path.join(path, file))
            models_info[label] = {"df": df, "color": color}
        except FileNotFoundError:
            print(f"Warning: File not found: {file}")
    return models_info


def plot_metrics(models_info, title_prefix,):
    # Plot R^2 scores:
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    })
    plt.figure(figsize=(3.5, 3.5))
    for label, info in models_info.items():
        df = info["df"]
        color = info["color"]
        plt.plot(df["epoch"],1-df["test_r2"], c=color, linestyle="-")
        # plt.plot(df["epoch"],1-df["train_r2"], c=color, linestyle="--", alpha=0.5)
        idx_max = np.argmax(df['test_r2'])
        print(f"{title_prefix} Aug: {label}, test R²: {df['test_r2'][idx_max]:.5f}, train R²: {df['train_r2'][idx_max]:.5f}, "
              f"test MSE: {df['test_mse'][idx_max]:.6f}, train MSE: {df['train_mse'][idx_max]:.6f}")
    legend_elements = [
        Line2D([0], [0], color=info["color"], lw=2, label=label)
        for label, info in models_info.items()
    ] + [
        Line2D([0], [0], color='black', linestyle='-', lw=2, label='Validation'),
        Line2D([0], [0], color='black', linestyle='--', lw=2, alpha=0.5, label='Train')
    ]
    plt.legend(handles=legend_elements, title="Run", frameon=False)
    plt.xlabel("Epochs")
    plt.ylabel(r"$1-R^2$")
    # plt.ylim(0.99, 1.001)
    # plt.xlim(left=110)
    plt.xscale('log')
    plt.yscale('log')
    # plt.xticks([100, 200, 300, 400, 500, 1000], [100, 200, 300, 400, 500,1000])
    plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    # plt.title(f"{title_prefix} Extra long run: 1-R²")
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"{title_prefix.lower()}_extralong_r2.pdf"), dpi=300, bbox_inches='tight')

def plot_metrics_vit(models_info, title_prefix,):
    # Plot R^2 scores:
    plt.figure(figsize=(3.5, 3.5))
    for label, info in models_info.items():
        df = info["df"]
        color = info["color"]
        plt.plot(1000*64*df["epoch"],1-df["test_r2"], c=color, linestyle="-")
        # plt.plot(1000*64*df["epoch"],1-df["train_r2"], c=color, linestyle="--", alpha=0.5)
        # plt.plot(df["epoch"],df["test_r2"], c=color, linestyle="-")
        # plt.plot(df["epoch"],df["train_r2"], c=color, linestyle="--", alpha=0.5)
        idx_max = np.argmax(df['test_r2'])
        print(f"{title_prefix} Aug: {label}, test R²: {df['test_r2'][idx_max]:.5f}, train R²: {df['train_r2'][idx_max]:.5f}, "
              f"test MSE: {df['test_mse'][idx_max]:.6f}, train MSE: {df['train_mse'][idx_max]:.6f}")
    legend_elements = [
        Line2D([0], [0], color=info["color"], lw=2, label=label)
        for label, info in models_info.items()
    ] + [
        Line2D([0], [0], color='black', linestyle='-', lw=2, label='Validation'),
        Line2D([0], [0], color='black', linestyle='--', lw=2, alpha=0.5, label='Train')
    ]
    plt.legend(handles=legend_elements, title="Model", frameon=False)
    plt.xlabel("Tokens processed")
    plt.ylabel(r"$1-R^2$ Score")
    # plt.ylim(0.99, 1.001)
    # plt.xlim(left=120, right=200)
    plt.xlim(left=1e6)
    plt.xscale('log')
    plt.yscale('log')
    # plt.xticks([100, 200, 300, 400, 500, 1000], [100, 200, 300, 400, 500,1000])
    plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    # plt.title(f"{title_prefix} Extra long run: 1-R²")
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"{title_prefix.lower()}_extralong_r2.pdf"), bbox_inches='tight')

run_fits_patch_sizes = {
    # "ViT-B16-500-epochs": "vit_b16_extralong.csv",
    "ViT-S16": "ViT_S16_500_epochs.csv",
    "ViT-T16": "ViT_T16_500_epochs.csv",
    "ViT-T8": "ViT_T8_all.csv",
    "ViT-S8": "ViT_S8_all.csv",
}
plot_metrics(load_models_info(run_fits_patch_sizes,"tab10"), "ViT Different Patch Sizes")


runs_vit_s16_diff_num_epochs = {
    "1k":"vit_s16_gradient_clip_long_test_3.csv",
    "600": "ViT_S16_600_epochs.csv",
    "500": "ViT_S16_500_epochs.csv",
    "400": "ViT_S16_400_epochs.csv",
    "300": "ViT_S16_300_epochs.csv",
    "200": "ViT_S16_200_epochs.csv",
    "100": "ViT_S16_100_epochs.csv",
}
# plot_metrics(load_models_info(runs_vit_s16_diff_num_epochs,"Reds"), "ViT-S16 Different Number of Epochs")

runs_convnexttiny = {
    "1k": "ConvNeXtTiny_all.csv",
    "600": "ConvNeXtTiny_600_epochs.csv",
    "500": "ConvNeXtTiny_500_epochs.csv",
    "400": "ConvNeXtTiny_400_epochs.csv",
    "300": "ConvNeXtTiny_300_epochs.csv",
    "200": "ConvNeXtTiny_200_epochs.csv",
    "100": "ConvNeXtTiny_100_epochs.csv",
}
# plot_metrics(load_models_info(runs_convnexttiny,"Blues"), "ConvNeXtTiny")

runs_convnextsmall = {
    "1k": "ConvNeXtSmall_all.csv",
    "500": "ConvNeXtSmall_500_epochs.csv",
    "400": "ConvNeXtSmall_400_epochs.csv",
    "300": "ConvNeXtSmall_300_epochs.csv",
    "200": "ConvNeXtSmall_200_epochs.csv",
    "100": "ConvNeXtSmall_100_epochs.csv",
}
# plot_metrics(load_models_info(runs_convnextsmall,"Greens"), "ConvNeXtSmall")

runs_vit_t16 = {
    "1k": "ViT_T16_all.csv",
    "600": "ViT_T16_600_epochs.csv",
    "500": "ViT_T16_500_epochs.csv",
    "400": "ViT_T16_400_epochs.csv",
    "300": "ViT_T16_300_epochs.csv",
    "200": "ViT_T16_200_epochs.csv",
    "100": "ViT_T16_100_epochs.csv",
}
# plot_metrics(load_models_info(runs_vit_t16,"Purples"), "ViT-T16")

runs_1000_epochs = {
    "ViT-B16-V1": "ViT_B16_1000_epochs_all_models_run.csv",
    # "ViT-B16-V2": "ViT_B16_1000_epochs_all_models_run_2.csv",
    # "ViT-B16-V3": "ViT_B16_all_models_run_3.csv",
    # "ViT-B16-V4": "ViT_B16_all_models_run_4.csv",
    # "ViT-B16": "vit_b16_extralong.csv",
    # "ConvNeXt-Small": "ConvNeXtSmall_all.csv",
    "ViT-S16": "vit_s16_gradient_clip_long_test_3.csv",
    # "ConvNeXt-Tiny": "ConvNeXtTiny_all.csv",
    # "ResNet-101": "ResNet101_all.csv",
    # "ResNet-50": "ResNet50_all.csv",
    "ViT-T16": "ViT_T16_all.csv",
}
plot_metrics_vit(load_models_info(runs_1000_epochs,"viridis"), "Scaling for 1000 Epochs")

def plot_grid_of_models(plot_groups, save_name="comparison_grid.pdf"):
    """
    plot_groups: list of (runs_dict, cmap, title) for each panel
                 should have length 4 for a 2x2 grid
    """
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    })
    fig, axes = plt.subplots(
        3, 2,
        figsize=(5.4, 6.4),
        sharex=True,
        sharey=True
    )
    axes = axes.flatten()

    for ax, (runs, cmap, title) in zip(axes, plot_groups):
        models_info = load_models_info(runs, cmap)
        for label, info in models_info.items():
            df = info["df"]
            ax.plot(df["epoch"], 1 - df["test_r2"], c=info["color"], label=label)
        ax.set_title(title, pad=6, fontsize=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
        # ax.legend(fontsize=11, frameon=True, title="Epochs")

    # Shared axis labels
    fig.text(0.5, 0.02, "Epochs", ha="center", fontsize=12)
    fig.text(0.02, 0.5, r"$1-R^2$", va="center", rotation="vertical", fontsize=13)

    # Collect legend handles only once
    # handles, labels = [], []
    # for ax in axes:
    #     h, l = ax.get_legend_handles_labels()
    #     handles.extend(h)
    #     labels.extend(l)

    # Global legend at bottom
    # fig.legend(
    #     handles, labels,
    #     loc="lower center", ncol=4, frameon=False,
    #     bbox_to_anchor=(0.5, -0.02)
    # )

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    plt.savefig(os.path.join(path, save_name), bbox_inches="tight")
    # plt.show()

runs_resnet50 = {
    "ResNet-50": "ResNet50_all.csv",
    # "500": "resnet50_gradient_clip_500_epochs.csv",
    "500": "ResNet50_500_epochs.csv",
    "400": "ResNet50_400_epochs.csv",
    "300": "ResNet50_300_epochs.csv",
    "200": "ResNet50_200_epochs.csv",
    "100": "resnet50_gradient_clip_test.csv",
}
runs_resnet101 = {
    "1000": "ResNet101_all.csv",
    "500": "ResNet101_500_epochs.csv",
    "400": "ResNet101_400_epochs.csv",
    "300": "ResNet101_300_epochs.csv",
    "200": "ResNet101_200_epochs.csv",
    "100": "resnet101_metrics_all_models_2.csv",
}


plot_groups = [
    (runs_vit_t16, "Reds", "ViT-T16"),
    (runs_vit_s16_diff_num_epochs, "Reds", "ViT-S16"),
    (runs_resnet50, "Greens", "ResNet-50"),
    (runs_resnet101, "Greens", "ResNet-101"),
    (runs_convnexttiny, "Blues", "ConvNeXt-Tiny"),
    (runs_convnextsmall, "Blues", "ConvNeXt-Small"),
]

plot_grid_of_models(plot_groups, save_name="all_models_row.pdf")


# runs_vit_s16_diff_num_datapoints = {
#     "ViT-S16-2k": "ViT_S16_2000_datapoints_run.csv",
#     "ViT-S16-4k": "ViT_S16_4000_datapoints_run.csv",
#     "ViT-S16-6k": "ViT_S16_6000_datapoints_run.csv",
#     "ViT-S16-8k": "ViT_S16_8000_datapoints_run.csv",
#     "ViT-S16-10k": "ViT_S16_10000_datapoints_run.csv",
#     "ViT-S16-12k": "ViT_S16_12000_datapoints_run.csv",
#     "ViT-S16-14k": "ViT_S16_14000_datapoints_run.csv",
#     "ViT-S16-16k": "ViT_S16_16000_datapoints_run.csv",
#     "ViT-S16-18k": "ViT_S16_18000_datapoints_run.csv",
#     "ViT-S16-20k": "ViT_S16_500_epochs.csv",
# }
# plot_metrics(load_models_info(runs_vit_s16_diff_num_datapoints,"Reds"), "ViT-S16 Different Number of Data Points")
# plot_datapoints_summary(load_models_info(runs_vit_s16_diff_num_datapoints,"Reds"), "ViT-S16 Different Number of Data Points")

# runs_convnextsmall_diff_num_datapoints = { 
#     "2000": "ConvNeXtSmall_2000_datapoints_run.csv",
#     "4000": "ConvNeXtSmall_4000_datapoints_run.csv",
#     "6000": "ConvNeXtSmall_6000_datapoints_run.csv",
#     "8000": "convnextsmall_8000_metrics_diff_num_datapoints.csv",
#     "10000": "convnextsmall_10000_metrics_diff_num_datapoints.csv",
#     "12000": "convnextsmall_12000_metrics_diff_num_datapoints.csv",
#     "14000": "convnextsmall_14000_metrics_diff_num_datapoints.csv",
#     "16000": "convnextsmall_16000_metrics_diff_num_datapoints.csv",
#     "18000": "convnextsmall_18000_metrics_diff_num_datapoints.csv",
#     "20000": "convnextsmall_metrics_all_models_2.csv",
# }
# plot_metrics(load_models_info(runs_convnextsmall_diff_num_datapoints,"Blues"), "ConvNeXtSmall Different Number of Data Points")

# runs_convnextsmall_diff_num_datapoints = { 
#     "None": "ViT_S16_none_dataaug_run.csv",
#     "H": "ViT_S16_hflip_dataaug_run.csv",
#     "V": "ViT_S16_vflip_dataaug_run.csv",
#     "R": "ViT_S16_rotate_dataaug_run.csv",
#     "H-V": "ViT_S16_hflip_vflip_dataaug_run.csv",
#     r"$D_4$": "ViT_S16_500_epochs.csv"
# }
# plot_metrics(load_models_info(runs_convnextsmall_diff_num_datapoints,"Blues"), "ViT-S16 Different Data Augmentation Techniques")

# runs_vit_t16_diff_num_datapoints = {
#         "2000": "ViT_T16_2000_datapoints_run.csv",
#         "4000": "ViT_T16_4000_datapoints_run.csv",
#         "6000": "ViT_T16_6000_datapoints_run.csv",
#         "8000": "ViT_T16_8000_datapoints_run.csv",
#         "10000": "ViT_T16_10000_datapoints_run.csv",
#         "12000": "ViT_T16_12000_datapoints_run.csv",
#         "14000": "ViT_T16_14000_datapoints_run.csv",
#         "16000": "ViT_T16_16000_datapoints_run.csv",
#         "18000": "ViT_T16_18000_datapoints_run.csv",
#         "20000": "ViT_T16_500_epochs.csv",
# }
# plot_metrics(load_models_info(runs_vit_t16_diff_num_datapoints,"Purples"), "ViT-T16 Different Number of Data Points")