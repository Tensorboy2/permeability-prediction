import os
import numpy as np
import torch
from data_loader import get_data
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib import gridspec

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

sns.set_theme(style="whitegrid")

from models.resnet import ResNet50, ResNet101
from models.vit import ViT_B16, ViT_S16, ViT_T16, ViT_T8, ViT_S8
from models.convnext import ConvNeXtTiny, ConvNeXtSmall

ROOT = os.path.join(os.path.dirname(__file__), "results")

models = {
    "ConvNeXtSmall": [ConvNeXtSmall, "ConvNeXtSmall_400_epochs.pth"],
    "ViT-T16": [ViT_T16,"ViT_T16_all.pth"],
    "ViT-S16": [ViT_S16,"ViT_S16.pth"],
    "ConvNeXtTiny": [ConvNeXtTiny,"ConvNeXtTiny_500_epochs.pth"],
    "ResNet50": [ResNet50,"ResNet50_500_epochs.pth"],
    "ResNet101": [ResNet101,"ResNet101_400_epochs.pth"],
    # "ViT-S8": [ViT_S8,"ViT_S8.pth"],
}

# Path for the combined cache file containing predictions for all models
CACHE_PATH = os.path.join(ROOT, "all_models_similarity_filled.npz")

preds = {}
if os.path.exists(CACHE_PATH):
    print("Loading existing combined predictions cache...")
    loaded = np.load(CACHE_PATH,mmap_mode='r', allow_pickle=True)
    # Each entry was saved as an object; convert back to dict
    preds = {model: loaded[model].item() for model in loaded.files}
    print(f"Loaded predictions for models: {list(preds.keys())}")
else:
    print("No combined cache found. Will generate predictions for all models.")

for model_name, (model_func, model_path) in models.items():
    # If we have cached results for this model, skip inference
    if model_name in preds:
        print(f"Skipping inference for {model_name} (cache found).")
        continue

    print(f"No existing predictions found for {model_name}. Generating new predictions.")
    # We'll load images and targets directly from the saved numpy/pt files instead
    image_path = "/data/test_images_filled.npy"
    targets_path = "/data/test_k.npy"

    print(f"Processing model: {model_name}, with weights from {model_path}")
    preds[model_name] = {}

    # Create model and move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_func(pre_trained_path=model_path)
    model.to(device)
    model.eval()

    # Load images
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    images = np.load(image_path)

    # Load target k files (support .npy and .pt). We expect each file to contain a (2,2) array
    targets = np.load(targets_path)
    # if len(target_files) == 0:
    #     # try to find a single targets file
    #     fallback = os.path.join(os.path.dirname(results_dir), 'test_k.npy')
    #     if os.path.exists(fallback):
    #         targets = np.load(fallback)
    #     else:
    #         raise FileNotFoundError(f"No target files found in {results_dir} and no fallback at {fallback}")
    # else:
    #     for fn in target_files:
    #         fp = os.path.join(results_dir, fn)
    #         if fn.endswith('.npy'):
    #             targets_list.append(np.load(fp))
    #         elif fn.endswith('.pt') or fn.endswith('.pth'):
    #             t = torch.load(fp)
    #             if isinstance(t, torch.Tensor):
    #                 targets_list.append(t.cpu().numpy())
    #             else:
    #                 targets_list.append(np.array(t))
    #         else:
    #             # try numpy load as fallback
    #             try:
    #                 targets_list.append(np.load(fp))
    #             except Exception:
    #                 print(f"Skipping unknown target file format: {fp}")

    #     targets = np.stack(targets_list, axis=0)

    # Normalize shapes to (N,2,2)
    images = np.asarray(images)
    targets = np.asarray(targets)
    if images.ndim == 3:
        # (N, H, W) -> add channel dim
        images = images[:, None, ...]

    # If targets are (N,4) flatten form, reshape
    if targets.ndim == 2 and targets.shape[1] == 4:
        targets = targets.reshape(-1, 2, 2)

    n_images = images.shape[0]
    n_targets = targets.shape[0]
    if n_images != n_targets:
        m = min(n_images, n_targets)
        print(f"Warning: number of images ({n_images}) != number of targets ({n_targets}). Truncating to {m}.")
        images = images[:m]
        targets = targets[:m]

    batch_size = 64
    all_preds = []
    with torch.no_grad():
        for i in range(0, images.shape[0], batch_size):
            batch = images[i:i+batch_size]
            batch_t = torch.from_numpy(batch).to(device).float()
            outputs = model(batch_t)
            out_np = outputs.cpu().numpy()
            # try to reshape to (B,2,2)
            try:
                out_np = out_np.reshape(out_np.shape[0], 2, 2)
            except Exception:
                # fallback: flatten last dim then reshape
                out_flat = out_np.reshape(out_np.shape[0], -1)
                if out_flat.shape[1] >= 4:
                    out_np = out_flat[:, :4].reshape(out_flat.shape[0], 2, 2)
                else:
                    raise ValueError("Model output cannot be reshaped to (B,2,2)")
            all_preds.append(out_np)

    preds_np = np.vstack(all_preds)
    preds[model_name]['image_filled'] = preds_np.squeeze()
    preds[model_name]['target'] = targets.squeeze()

    # Compute per-component R^2 and mean R^2
    y_true = preds[model_name]['target'].reshape(-1, 2, 2)
    y_pred = preds[model_name]['image_filled'].reshape(-1, 2, 2)
    r2 = np.zeros((2, 2), dtype=float)
    for ii in range(2):
        for jj in range(2):
            yt = y_true[:, ii, jj]
            yp = y_pred[:, ii, jj]
            ss_res = np.sum((yt - yp) ** 2)
            ss_tot = np.sum((yt - yt.mean()) ** 2)
            if ss_tot == 0.0:
                r2[ii, jj] = 1.0 if ss_res == 0.0 else 0.0
            else:
                r2[ii, jj] = 1.0 - ss_res / ss_tot

    preds[model_name]['r2'] = r2
    preds[model_name]['r2_mean'] = float(np.mean(r2))

    print(f"Completed model: {model_name}  mean R^2={preds[model_name]['r2_mean']:.4f}")

    # Save combined cache after each model so partial results are preserved
    np.savez_compressed(CACHE_PATH, **{k: v for k, v in preds.items()})
    print(f"Saved combined cache with models: {list(preds.keys())}")


def plot_model_scatter(preds, model_name, cmap="viridis"):
    """
    Makes a 2x2 scatter plot for predictions vs targets using seaborn.
    Each subplot corresponds to one component of k.
    Includes a shared colorbar and unified color scale across all subplots.
    """
    # Print model name and R^2 scores
    r2 = preds[model_name]['r2']
    r2_mean = preds[model_name]['r2_mean']
    print(f"Plotting {model_name}  mean R^2={r2_mean:.5f}")

    fig, ax = plt.subplots(2, 2, figsize=(6.4, 6.4))

    # === 1. Compute global error range across all components ===
    errors = []
    for i in range(2):
        for j in range(2):
            y_true = preds[model_name]["target"][:, i, j]
            y_pred = preds[model_name]["image_filled"][:, i, j]
            errors.append(np.abs(y_pred - y_true))
    errors = np.concatenate(errors)
    norm = Normalize(vmin=errors.min(), vmax=errors.max())

    # === 2. Scatter plots ===
    for i in range(2):
        for j in range(2):
            y_true = preds[model_name]["target"][:, i, j]
            y_pred = preds[model_name]["image_filled"][:, i, j]
            error = np.abs(y_pred - y_true)

            sns.scatterplot(
                x=y_true, y=y_pred,
                hue=error,
                palette=cmap,
                hue_norm=norm,        # <-- unified color scale
                s=10, alpha=0.7,
                legend=False,
                ax=ax[i, j]
            )

            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax[i, j].plot([min_val, max_val], [min_val, max_val],
                          '-', lw=1, alpha=0.4, color='red')

            sub = f"{'x' if i == 0 else 'y'}{'x' if j == 0 else 'y'}"
            ax[i, j].set_title(f"$k_{{{sub}}}$", fontsize=14)
            ax[i, j].set_xlabel("Target")
            ax[i, j].set_ylabel("Prediction")
            ax[i, j].grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    # === 3. Shared colorbar ===
    plt.tight_layout()  # leave space on right for colorbar
    plt.subplots_adjust(right=0.8)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # dummy
    cbar_ax = fig.add_axes([0.83, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
    fig.colorbar(sm, cax=cbar_ax, shrink=0.8, label="Absolute Error")
    

    # plt.tight_layout()
    plt.savefig(os.path.join(ROOT, f"{model_name}_similarity_filled.pdf"), dpi=300)
    plt.close()

# cmaps = ["viridis","viridis","viridis","viridis","viridis","viridis"]
# for model_name,cmap in zip(models.keys(), cmaps):
#     plot_model_scatter(preds, model_name, cmap=cmap)


def make_KC_model(test_target):
    """
    Fit Kozeny-Carman C parameter for each component and return a 2x2 matrix of C values.
    Uses only training data.
    
    Kozeny-Carman equation: k = C * φ^3 / (1-φ)^2
    Where:
        k = permeability
        φ = porosity
        C = Kozeny-Carman constant (fitted per component)
    """
    train_size = 16000
    data_path = os.path.join(os.path.dirname(__file__), "data")
    train_and_val_images = np.load(os.path.join(data_path, 'images_filled.npy'),mmap_mode='r')[:train_size]
    permeabilities = np.load(os.path.join(data_path, 'train_and_val_k.npy'),mmap_mode='r')[:train_size]
    test_image_path = os.path.join(data_path, "test_images_filled.npy")
    test_images = np.load(test_image_path,mmap_mode='r')
    # Training porosities (solid fraction, so porosity = 1 - solid)
    train_porosities = 1 - train_and_val_images.mean(axis=(1,2))  # shape (N,)
   
    KC_predictions = np.zeros_like(test_target)
    
    for i in range(2):
        for j in range(2):
            if i!=j:
                KC_predictions[:,i,j] = np.zeros_like(test_poro)#*C_median * (test_poro**3) / ((1 - test_poro)**2)
            else:    
                k_values = permeabilities[:,i,j]
                
                # Filter for valid porosity range and finite k values
                mask = (train_porosities > 0) & (train_porosities < 1) & np.isfinite(k_values)
                
                # Fit C: rearrange k = C * φ³/(1-φ)² to get C = k * (1-φ)²/φ³
                C_vals = k_values[mask] * (1 - train_porosities[mask])**2 / (train_porosities[mask]**3)
                C_vals = C_vals[np.isfinite(C_vals) & (C_vals > 0)]
                C_median = np.median(C_vals)
                
                print(f"Fitted C[{i},{j}] = {C_median:.4e}")
                
                # Predict using fitted C: k = C * φ³/(1-φ)²
                test_poro = 1- test_images.mean(axis=(1,2))  # avoid zero porosity
                
                # Handle edge cases where porosity is 0 or 1
                # safe_poro = np.clip(test_poro, 1e-6, 1 - 1e-6)
                KC_predictions[:,i,j] = C_median * (test_poro**3) / ((1 - test_poro)**2)
            # KC_predictions[:,i,j] = C_median * (safe_poro**3) / ((1 - safe_poro)**2)

    # Compute R^2 for the KC model
    y_true = test_target.reshape(-1, 2, 2)
    y_pred = KC_predictions.reshape(-1, 2, 2)
    r2 = np.zeros((2, 2), dtype=float)
    for ii in range(2):
        for jj in range(2):
            yt = y_true[:, ii, jj]
            yp = y_pred[:, ii, jj]
            ss_res = np.sum((yt - yp) ** 2)
            ss_tot = np.sum((yt - yt.mean()) ** 2)
            if ss_tot == 0.0:
                r2[ii, jj] = 1.0 if ss_res == 0.0 else 0.0
            else:
                r2[ii, jj] = 1.0 - ss_res / ss_tot
    r2_mean = float(np.mean(r2))
    print(f"Kozeny-Carman model mean R^2={r2_mean:.5f}")

    return KC_predictions


def plot_combined_KC_models(preds, models_to_plot):
    """
    Combined figure:
    - Col1: Kozeny-Carman merged plot (row1: diag, row2: offdiag)
    - Col2: ViT-S8
    - Col3: ConvNeXt-Small
    """
    # test_images = preds["ConvNeXtSmall"]['image_filled']

    test_target = preds["ConvNeXtSmall"]['target']
    KC_preds = make_KC_model(test_target)  # user-defined
    sns.set_theme(style="whitegrid")

    # --- Figure setup ---
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    })
    fig, axes = plt.subplots(
        2, 3,
        figsize=(7, 3.5),
        width_ratios=[1, 1, 1],
        height_ratios=[1, 1],
        # constrained_layout=True,
    )

    # --- Column 1: Kozeny–Carman model ---
    # porosity_values = 1 - test_images.mean(axis=(1, 2))

    merged = [
        ("Diagonal", [(0, 0), (1, 1)]),
        ("Off-diagonal", [(0, 1), (1, 0)]),
    ]

    for row, (label, pairs) in enumerate(merged):
        ax = axes[row, 0]
        ax.grid(True)

        # Plot all (i, j) pairs for this row
        for (i, j) in pairs:
            ax.scatter(
                test_target[:, i, j],
                KC_preds[:, i, j],
                alpha=0.2,
                s=5,
                edgecolors="black",
                facecolor='none',
            )

        # Reference line (min/max computed from the plotted points)
        all_true = np.concatenate([test_target[:, i, j] for (i, j) in pairs])
        all_pred = np.concatenate([KC_preds[:, i, j] for (i, j) in pairs])
        min_val = min(all_true.min(), all_pred.min())
        max_val = max(all_true.max(), all_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r-", alpha=0.4)
        # ax.set_xscale('log')
        # ax.set_yscale('log')

        # Axis labels and titles
        if row == 0:
            ax.set_title("Kozeny–Carman")
            ax.set_ylabel(r"Prediction ($K_{xx}, K_{yy}$)")
        else:
            ax.set_xlabel("Target")
            ax.set_ylabel(r"Prediction ($K_{xy}, K_{yx}$)")

    # --- Columns 2 & 3: Learned models ---
    for col, model_name in enumerate(models_to_plot, start=1):
        y_true = preds[model_name]['target']
        y_pred = preds[model_name]['image_filled']

        # Two rows: diagonal and off-diagonal
        merged = [
            ("Diagonal", [(0, 0), (1, 1)]),
            ("Off-diagonal", [(0, 1), (1, 0)]),
        ]

        for row, (label, pairs) in enumerate(merged):
            ax = axes[row, col]

            # Plot both (i,j) pairs in this category
            for (i, j) in pairs:
                sns.scatterplot(
                    x=y_true[:, i, j],
                    y=y_pred[:, i, j],
                    edgecolor="black",
                    s=5,
                    alpha=0.2,
                    legend=False,
                    ax=ax,
                    facecolor='none',
                )

            # Reference line
            all_true = np.concatenate([y_true[:, i, j] for (i, j) in pairs])
            all_pred = np.concatenate([y_pred[:, i, j] for (i, j) in pairs])
            min_val = min(all_true.min(), all_pred.min())
            max_val = max(all_true.max(), all_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], "r-", alpha=0.4)
            # ax.set_xscale('log')
            # ax.set_yscale('log')
            # Labels and formatting
            ax.grid(True)
            if row == 0:
                ax.set_title(f"{model_name}")
                # ax.set_ylabel(r"Prediction ($K_{xx}, K_{yy}$)")
            else:
                ax.set_xlabel("Target")
                # ax.set_ylabel(r"Prediction ($K_{xy}, K_{yx}$)")
                # ax.set_xticks(ticks=(),fontsize=7)
                # ax.set_yticks(fontsize=7)
    # --- Final layout and save ---
    plt.tight_layout()
    plt.savefig("combined_KC_models.pdf", dpi=300)
    plt.close()

plot_combined_KC_models(preds, models_to_plot=["ViT-T16","ConvNeXtSmall"])