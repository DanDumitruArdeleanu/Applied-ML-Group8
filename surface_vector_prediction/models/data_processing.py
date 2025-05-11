import os
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pyminiply


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# Configuration
models_dir = Path("models")
output_dir = ensure_dir("data_analysis")
model_files = sorted(models_dir.glob("obj_*.ply"))
labels = [f"Obj_{i+1:02d}" for i in range(len(model_files))]

# Storage for results
results = {label: {} for label in labels}
aggregate_features = []
aggregate_labels = []

# Process each model
def process_model(filepath, idx):
    label = labels[idx]
    msgs = []
    try:
        verts, _, normals, _, colors = pyminiply.read(filepath)
        n = len(verts) if verts is not None else 0
        results[label]["vertex_count"] = n
        if n > 0:
            # Handle normals
            if isinstance(normals, np.ndarray) and normals.shape == (n, 3):
                norms = np.linalg.norm(normals, axis=1, keepdims=True)
                zero_mask = norms.flatten() == 0
                normals_proc = normals.copy()
                if np.any(zero_mask):
                    normals_proc[zero_mask] = 0
                    norms[zero_mask.reshape(-1, 1)] = 1.0
                normals_norm = normals_proc / norms
                # Combine features and remove NaNs
                feats = np.hstack((verts, normals_norm))
                feats = np.nan_to_num(feats)
                # PCA if possible
                if feats.shape[1] >= 2:
                    pca = PCA(n_components=min(feats.shape))
                    pca_res = pca.fit_transform(feats)
                    results[label].update(
                        pca_results=pca_res,
                        explained_variance=pca.explained_variance_ratio_,
                    )
                    aggregate_features.append(feats)
                    aggregate_labels.append(np.full(n, idx, dtype=int))
                msgs.append("PCA done")
            else:
                msgs.append("No valid normals")
            # Normalize colors if present
            if isinstance(colors, np.ndarray) and colors.shape[0] == n:
                results[label]["colors_norm"] = colors.astype(np.float32) / 255.0
        else:
            msgs.append("No vertices")
    except Exception as e:
        msgs.append(f"Error: {e}")
    print(f"{filepath.name}: {', '.join(msgs)}")


# Run processing
for idx, file in enumerate(model_files):
    process_model(file, idx)

# Global PCA
if aggregate_features:
    all_feats = np.vstack(aggregate_features)
    all_idxs = np.concatenate(aggregate_labels)
    n_comp = min(6, all_feats.shape[0], all_feats.shape[1])
    if n_comp >= 2:
        global_pca = PCA(n_components=n_comp)
        global_res = global_pca.fit_transform(all_feats)
        var = global_pca.explained_variance_ratio_
        # Plot global PCA
        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = plt.get_cmap("tab20")
        scatter = ax.scatter(
            global_res[:, 0], global_res[:, 1], c=all_idxs, cmap=cmap, s=1, alpha=0.4
        )
        ax.set(title="Global PCA (Vertices + Normals)", xlabel="PC1", ylabel="PC2")
        txt = "\n".join([f"PC{i+1}: {v:.3f}" for i, v in enumerate(var[:6])])
        ax.text(
            0.02,
            0.02,
            txt,
            transform=ax.transAxes,
            bbox=dict(facecolor="wheat", alpha=0.6),
        )
        legend_elems = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=cmap(i),
                label=labels[i],
                linestyle="",
            )
            for i in np.unique(all_idxs)
        ]
        ax.legend(
            handles=legend_elems,
            title="Models",
            bbox_to_anchor=(1.04, 1),
            loc="upper left",
        )
        fig.tight_layout(rect=[0, 0, 0.85, 1])
        fig.savefig(output_dir / "global_pca.png", dpi=150)
        plt.close(fig)

# Vertex count bar plot
fig, ax = plt.subplots(figsize=(12, 7))
counts = [results[l].get("vertex_count", 0) for l in labels]
bars = ax.bar(labels, counts)
ax.set(title="Vertex Counts per Model", xlabel="Model", ylabel="Count")
ax.bar_label(bars)
plt.xticks(rotation=60)
fig.tight_layout()
fig.savefig(output_dir / "vertex_counts.png", dpi=150)
plt.close(fig)

# Individual model plots
def plot_individual(label, info, idx):
    n = info.get("vertex_count", 0)
    if n <= 0:
        return
    # Coordinate histograms
    verts = pyminiply.read(models_dir / f"obj_{idx+1:02d}.ply")[0]
    if verts is not None and verts.shape[1] >= 3:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
        for j, ax in enumerate(axs):
            ax.hist(verts[:, j], bins=50, alpha=0.8)
            ax.set(title=["X", "Y", "Z"][j], xlabel="Value")
        fig.suptitle(f"Coordinate Distributions ({label})")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(output_dir / f"{label}_coords.png", dpi=120)
        plt.close(fig)
    # PCA scatter
    pca_res = info.get("pca_results")
    if pca_res is not None and pca_res.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(8, 7))
        color_data = info.get("colors_norm", pca_res[:, 0])
        scatter = ax.scatter(
            pca_res[:, 0],
            pca_res[:, 1],
            c=color_data,
            s=5,
            alpha=0.7,
            cmap=None if "colors_norm" in info else "viridis",
        )
        ax.set(title=f"PCA ({label})", xlabel="PC1", ylabel="PC2")
        fig.tight_layout()
        fig.savefig(output_dir / f"{label}_pca.png", dpi=150)
        plt.close(fig)


for idx, lbl in enumerate(labels):
    plot_individual(lbl, results[lbl], idx)

print("Analysis complete. Files saved to", output_dir)
