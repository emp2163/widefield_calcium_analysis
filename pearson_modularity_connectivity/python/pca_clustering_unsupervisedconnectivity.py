#!/usr/bin/env python3
# ============================================================
# Unsupervised Connectivity: PCA + Hierarchical Clustering
#  (Genotype-grouped, streaming, parallel, spawn-safe)
#
# Adds statistical comparisons across genotypes:
#   1) Label similarity @ fixed k (ARI/NMI/VI/FM + optimal alignment) + trial-level permutation p-values
#   2) Dendrogram shape difference (Mantel on cophenetic distances) + permutation p-values
#   3) Block-structure strength (within–between correlation gap) with bootstrap CIs and gap-difference test
# ============================================================

# =========================
# 1) IMPORTS
# =========================
import os, glob, pickle
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr
from scipy.cluster.hierarchy import linkage, dendrogram, optimal_leaf_ordering, fcluster, cophenet
from scipy.spatial.distance import squareform, pdist
import multiprocessing as mp
from contextlib import contextmanager
import itertools as it

# Metrics
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
from scipy.stats import spearmanr

rng_global = np.random.default_rng(0)

# Optional: nicer default DPI
try:
    import ISLP  # noqa: F401
    plt.rcParams.update({"figure.dpi": 140})
except Exception:
    pass

# =========================
# 2) CONFIG
# =========================

# Excluded ROIs
EXCLUDED_ROIS = ['L-ACA', 'R-ACA', 'L-PL', 'R-PL', 'L-RSPv', 'R-RSPv']

# Phases definition in samples (20Hz sampling, spectrogram time starts at -5s)
PHASES = {
    'pre':    (0, 120),   # -5 to +1 sec
    'during': (80, 220),  # -1 to +6 sec
    'post':   (180, 320)  # +4 to +11 sec
}

# Clustering knobs
N_CLUSTERS_LIST = (4, 6)
LINKAGES = ("average", "complete")
K_PCS_FOR_WARD = 6              # enforce k=6 PCs for Ward/PCA clustering
K_TARGET_VARIANCE = 0.80        # guard for k* selection

# ===== Results dir (guard print so workers don't echo) =====
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = f'/Users/ep/Desktop/Pearsons/pearson_analysis_results_pca_clustering_{timestamp}'
os.makedirs(RESULTS_DIR, exist_ok=True)
if mp.current_process().name == "MainProcess":
    print(f"[INFO] Saving outputs to: {RESULTS_DIR}")

# =========================
# 3) FILE / PATH HELPERS
# =========================
def list_trial_paths(folder: str, recursive: bool = True):
    if os.path.isdir(folder):
        pkl_paths = sorted(glob.glob(os.path.join(folder, "*.pkl")))
        base = folder
    else:
        if not recursive:
            raise FileNotFoundError(f"Folder not found: {folder}")
        parent = os.path.dirname(folder)
        sub = os.path.basename(folder)
        search_root = parent if os.path.isdir(parent) else os.path.expanduser("~")
        pkl_paths = sorted(glob.glob(os.path.join(search_root, "**", sub, "**", "*.pkl"),
                                     recursive=True))
        base = os.path.commonpath(pkl_paths) if pkl_paths else folder

    if not pkl_paths:
        raise FileNotFoundError(
            f"No .pkl trial files found.\nLooked in: {folder}\n"
            f"(and recursively under {os.path.dirname(folder)} if needed)"
        )
    if mp.current_process().name == "MainProcess":
        print(f"[INFO] Found {len(pkl_paths)} trial files under {base}")
    return pkl_paths


def read_trial(pkl_path: str):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _read_meta_field(pkl_path: str, field: str, default=None):
    tr = read_trial(pkl_path)
    return tr.get("metadata", {}).get(field, default)


def partition_paths_by_genotype(trial_paths: list):
    groups = {}
    for p in trial_paths:
        g = _read_meta_field(p, "genotype", default=None)
        if g is None:
            continue
        groups.setdefault(str(g), []).append(p)
    return {k: sorted(v) for k, v in sorted(groups.items(), key=lambda kv: kv[0])}

# =========================
# 4) CONNECTIVITY BUILDERS
# =========================

def trial_roi_pair_corr(roi_data_dict: dict):
    rois_all = sorted(roi_data_dict.keys())
    rois = [r for r in rois_all if r not in EXCLUDED_ROIS]
    n = len(rois)
    z_mats = {phase: np.full((n, n), np.nan, dtype=np.float32) for phase in PHASES}

    for i in range(n):
        Ai = roi_data_dict[rois[i]]
        for j in range(i, n):
            Bj = roi_data_dict[rois[j]]
            for phase, (a, b) in PHASES.items():
                A = Ai[:, a:b].flatten()
                B = Bj[:, a:b].flatten()
                if A.size > 1 and np.all(np.isfinite(A)) and np.all(np.isfinite(B)):
                    r, _ = pearsonr(A, B)
                    if np.isfinite(r):
                        r = np.clip(r, -0.999999, 0.999999)
                        z = np.arctanh(r)
                        z_mats[phase][i, j] = z
                        z_mats[phase][j, i] = z

    for phase in z_mats:
        np.fill_diagonal(z_mats[phase], 0.0)
        z_mats[phase] = z_mats[phase].astype(np.float32, copy=False)
    return rois, z_mats


# =========================
# 5) PARALLEL STREAMING COLLECTOR (SPAWN-SAFE)
# =========================

def _worker_compute_contrib(pkl_path, rois_expected):
    try:
        tr = read_trial(pkl_path)
        rois, z_mats = trial_roi_pair_corr(tr['roi_spectrograms'])
        if rois_expected is not None and rois != rois_expected:
            return (False, rois, None, None)
        sumZ, cntZ = {}, {}
        for phase, Z in z_mats.items():
            Zf = Z.astype(np.float32, copy=False)
            mask = np.isfinite(Zf)
            sumZ[phase] = Zf
            c = np.zeros_like(Zf, dtype=np.int32); c[mask] = 1
            cntZ[phase] = c
        return (True, rois, sumZ, cntZ)
    except Exception:
        return (False, None, None, None)

def _worker_compute_contrib_star(args):
    pkl_path, rois_expected = args
    return _worker_compute_contrib(pkl_path, rois_expected)

def _yield_batches(seq, batch_size):
    for i in range(0, len(seq), batch_size):
        yield seq[i:i+batch_size]

@contextmanager
def _mp_pool(n_workers):
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=n_workers)
    try:
        yield pool
    finally:
        pool.close(); pool.join()

def collect_all_z_parallel(trial_paths: list,
                           batch_size: int = 256,
                           n_workers: int | None = None,
                           longform_csv_path: str | None = None,
                           write_longform: bool = False):
    assert len(trial_paths) > 0, "No trial paths provided."

    first = read_trial(trial_paths[0])
    rois_ref, _ = trial_roi_pair_corr(first['roi_spectrograms'])
    n = len(rois_ref)

    sumZ = {p: np.zeros((n, n), dtype=np.float32) for p in PHASES}
    cntZ = {p: np.zeros((n, n), dtype=np.int64)  for p in PHASES}

    lf = None
    if write_longform and longform_csv_path:
        lf = open(longform_csv_path, "w"); lf.write("phase,roi1,roi2,z\n")

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    if mp.current_process().name == "MainProcess":
        print(f"[PAR] Using {n_workers} workers; batch_size={batch_size}; trials={len(trial_paths)}")

    iu, ju = np.triu_indices(n, 1)
    total_batches = int(np.ceil(len(trial_paths) / batch_size))
    for bi, batch in enumerate(_yield_batches(trial_paths, batch_size)):
        if mp.current_process().name == "MainProcess":
            print(f"    [PAR] Batch {bi+1}/{total_batches} (size={len(batch)})")
        with _mp_pool(n_workers) as pool:
            iterable = it.zip_longest(batch, [], fillvalue=rois_ref)
            for ok, rois, sZ, cZ in pool.imap_unordered(_worker_compute_contrib_star, iterable, chunksize=8):
                if not ok:
                    if rois is not None and rois != rois_ref:
                        raise RuntimeError("ROI order differs across trials after exclusions.")
                    continue
                for phase in PHASES:
                    sumZ[phase] += sZ[phase]
                    cntZ[phase] += cZ[phase]
                if lf is not None:
                    for phase in PHASES:
                        z_edges = sZ[phase][iu, ju]
                        keep = np.isfinite(z_edges)
                        for i_idx, j_idx, zval in zip(iu[keep], ju[keep], z_edges[keep]):
                            lf.write(f"{phase},{rois_ref[i_idx]},{rois_ref[j_idx]},{float(zval):.6f}\n")
                del sZ, cZ
    if lf is not None:
        lf.close()

    Z_mean, R_mean = {}, {}
    for phase in PHASES:
        with np.errstate(invalid='ignore', divide='ignore'):
            Zm = sumZ[phase] / np.maximum(cntZ[phase], 1)
        Zm[np.isnan(Zm)] = 0.0
        np.fill_diagonal(Zm, 0.0)
        Z_mean[phase] = Zm.astype(np.float32, copy=False)
        R_mean[phase] = np.tanh(Z_mean[phase].astype(np.float64)).astype(np.float32)

    return rois_ref, Z_mean, R_mean

# =========================
# 6) PCA HELPERS + METRICS
# =========================

def pca_on_connectivity_profiles(Z_mean: np.ndarray, var_keep=None, random_state=0):
    X = Z_mean.copy()
    np.fill_diagonal(X, 0.0)
    Xc = X - X.mean(axis=0, keepdims=True)

    n_components = None
    if isinstance(var_keep, float):
        if 0.0 < var_keep < 1.0:
            n_components = var_keep
        elif var_keep >= 1.0:
            n_components = None
    elif isinstance(var_keep, int):
        n_components = var_keep if var_keep >= 1 else None
    else:
        n_components = None

    pca = PCA(n_components=n_components, svd_solver='full')
    scores = pca.fit_transform(Xc)
    return pca, scores, Xc


def broken_stick_expectation(m: int):
    hs = np.array([np.sum(1.0 / np.arange(k, m + 1)) for k in range(1, m + 1)])
    bs = hs / m
    return bs


def parallel_analysis(Xc: np.ndarray, n_perm: int = 200, random_state: int = 0):
    rng = np.random.default_rng(random_state)
    n, p = Xc.shape
    evals_accum = np.zeros(p, dtype=np.float64)
    for _ in range(n_perm):
        Xp = Xc.copy()
        for j in range(p):
            rng.shuffle(Xp[:, j])
        pca = PCA(n_components=None, svd_solver='full')
        pca.fit(Xp)
        ev = pca.explained_variance_
        evals_accum[:len(ev)] += ev
    return evals_accum / n_perm


def reconstruction_R2(Xc: np.ndarray, pca: PCA):
    X = Xc
    U = pca.transform(X)   # scores
    Vt = pca.components_
    X_norm2 = np.sum(X * X)
    R2 = []
    X_hat = np.zeros_like(X)
    for k in range(1, Vt.shape[0] + 1):
        X_hat += np.outer(U[:, k-1], Vt[k-1, :])
        num = np.sum((X - X_hat)**2)
        R2.append(1.0 - num / X_norm2)
    return np.array(R2)


def _align_sign(a, b):
    return a if np.dot(a, b) >= 0 else -a


def bootstrap_pc_stability(Xc: np.ndarray, n_boot: int = 200, random_state: int = 0, k_check=(1,2)):
    rng = np.random.default_rng(random_state)
    ref_pca = PCA(n_components=None, svd_solver='full')
    ref_pca.fit(Xc)
    ref_load = ref_pca.components_

    out = {}
    for pc_index in k_check:
        i = pc_index - 1
        ref = ref_load[i, :]
        corrs = []
        for _ in range(n_boot):
            idx = rng.integers(0, Xc.shape[0], size=Xc.shape[0])
            Xb = Xc[idx, :]
            p = PCA(n_components=None, svd_solver='full')
            p.fit(Xb)
            if i >= p.components_.shape[0]:
                continue
            test = _align_sign(p.components_[i, :], ref)
            c = np.corrcoef(ref, test)[0, 1]
            corrs.append(abs(c))
        out[pc_index] = np.median(corrs) if corrs else np.nan
    return out


def within_between_block_score(R: np.ndarray, labels: np.ndarray):
    n = len(labels)
    within_vals, between_vals = [], []
    for i in range(n):
        for j in range(i+1, n):
            if labels[i] == labels[j]:
                within_vals.append(R[i, j])
            else:
                between_vals.append(R[i, j])
    w = np.mean(within_vals) if within_vals else 0.0
    b = np.mean(between_vals) if between_vals else 0.0
    return float(w - b)


def weighted_pc_score(scores: np.ndarray, evr: np.ndarray, k_star: int):
    k = min(k_star, scores.shape[1])
    w = evr[:k] / (evr[:k].sum() + 1e-12)
    return (scores[:, :k] * w[None, :]).sum(axis=1)

# =========================
# 7) PLOTTING HELPERS
# =========================

def plot_explained_variance(pca, title: str, outpath: str):
    evr = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(np.arange(1, len(evr)+1), evr, edgecolor='k', label='Per-PC variance')
    ax.plot(np.arange(1, len(evr)+1), np.cumsum(evr), marker='o', lw=2, label='Cumulative')
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)

def plot_pca_scatter(scores: np.ndarray, roi_labels, pcx: int, pcy: int, title: str, outpath: str):
    i, j = pcx-1, pcy-1
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    ax.scatter(scores[:, i], scores[:, j], s=30)
    for k, r in enumerate(roi_labels):
        ax.text(scores[k, i], scores[k, j], r, fontsize=8, ha='left', va='center', clip_on=True)
    ax.axhline(0, lw=0.8, ls=':', color='k'); ax.axvline(0, lw=0.8, ls=':', color='k')
    ax.set_xlabel(f"PC{pcx}"); ax.set_ylabel(f"PC{pcy}")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220); plt.close(fig)

def plot_loadings_heatmap(pca, roi_labels, k_first: int, title: str, outpath: str):
    W = pca.components_[:k_first, :]
    plt.figure(figsize=(7.8, min(0.28*len(roi_labels)+2.5, 9)))
    sns.heatmap(W, cmap="coolwarm", center=0, xticklabels=roi_labels,
                yticklabels=[f"PC{i+1}" for i in range(k_first)])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()

def plot_top_loading_bars(pca, roi_labels, pc_index: int, top_n: int, title: str, outpath: str):
    i = pc_index - 1
    load = pca.components_[i, :]
    idx = np.argsort(np.abs(load))[::-1][:top_n]
    labs = [roi_labels[k] for k in idx]
    vals = load[idx]
    fig, ax = plt.subplots(figsize=(7.6, 0.35*top_n + 1.5))
    y = np.arange(len(idx))
    ax.barh(y, vals); ax.set_yticks(y); ax.set_yticklabels(labs); ax.invert_yaxis()
    ax.set_xlabel("Loading"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(outpath, dpi=220); plt.close(fig)

def export_pca_tables(pca, scores, roi_labels, out_prefix: str):
    evr = pca.explained_variance_ratio_
    df_var = pd.DataFrame({
        "PC": np.arange(1, len(evr)+1),
        "explained_var": evr,
        "explained_var_cum": np.cumsum(evr)
    })
    df_var.to_csv(out_prefix + "_pca_variance.csv", index=False)

    cols_scores = [f"PC{i}" for i in range(1, scores.shape[1]+1)]
    df_scores = pd.DataFrame(scores, columns=cols_scores)
    df_scores.insert(0, "roi", roi_labels)
    df_scores.to_csv(out_prefix + "_pca_scores.csv", index=False)

    loadings = pca.components_.T
    cols_load = [f"PC{i}" for i in range(1, loadings.shape[1]+1)]
    df_load = pd.DataFrame(loadings, columns=cols_load)
    df_load.insert(0, "roi", roi_labels)
    df_load.to_csv(out_prefix + "_pca_loadings.csv", index=False)

def plot_dendro(Z, labels, title, outpath):
    plt.figure(figsize=(9, 4))
    dendrogram(Z, labels=labels, leaf_rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()

def heatmap_reordered(M, order, labels, title, outpath, vmin=-1, vmax=1):
    M_ord = M[np.ix_(order, order)]
    labs = [labels[i] for i in order]
    plt.figure(figsize=(6.8, 6.0))
    sns.heatmap(M_ord, cmap="coolwarm", vmin=vmin, vmax=vmax, center=0,
                xticklabels=labs, yticklabels=labs, square=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()

def corr_distance_from_R(R: np.ndarray) -> np.ndarray:
    R = np.clip(R, -1.0, 1.0)
    D = np.sqrt(2.0 * (1.0 - R))
    np.fill_diagonal(D, 0.0)
    return D

# =========================
# 8) CLUSTERING ROUTES
# =========================

def cluster_route_A(R, roi_labels, linkage_method="average", k=6):
    D = corr_distance_from_R(R)
    y = squareform(D, checks=False)
    Z = linkage(y, method=linkage_method)
    Z_olo = optimal_leaf_ordering(Z, y)
    order = dendrogram(Z_olo, no_plot=True)['leaves']
    labels = fcluster(Z_olo, t=k, criterion='maxclust')
    return order, labels, Z_olo

def cluster_route_B_Ward_PCA_fixed_k(Z_mean, roi_labels, k_pcs: int, k_clusters: int = 6):
    pca, scores, _ = pca_on_connectivity_profiles(Z_mean, var_keep=None, random_state=0)
    k_use = min(k_pcs, scores.shape[1])
    scores_k = scores[:, :k_use]
    Z = linkage(scores_k, method='ward')
    order = dendrogram(Z, no_plot=True)['leaves']
    labels = fcluster(Z, t=k_clusters, criterion='maxclust')
    return order, labels, Z, pca, scores_k

# =========================
# 9) RUNNERS (PER GENOTYPE GROUP)
# =========================

def run_condition_for_group(condition_name: str,
                            group_name: str,
                            trial_paths: list,
                            results_dir: str,
                            n_clusters_list=N_CLUSTERS_LIST,
                            linkages=LINKAGES,
                            k_target_variance=K_TARGET_VARIANCE,
                            n_boot_pc_stability=200,
                            batch_size=256,
                            n_workers=None,
                            random_state=0,
                            fast_mode: bool = False,
                            write_artifacts: bool = True):
    tag = f"{condition_name}_{group_name}"
    longform_csv_path = (os.path.join(results_dir, f"{tag}_pairs_longform.csv")
                         if (write_artifacts) else None)

    roi_labels, Z_mean, R_mean = collect_all_z_parallel(
        trial_paths=trial_paths,
        batch_size=batch_size,
        n_workers=n_workers,
        longform_csv_path=longform_csv_path,
        write_longform=(write_artifacts and False)  # keep off unless you really need it
    )

    all_label_tables = []
    all_cluster_metrics = []

    for phase in PHASES.keys():
        R = R_mean[phase]; Zm = Z_mean[phase]
        prefix = os.path.join(results_dir, f"{tag}_{phase}")

        # PCA + thresholds
        pca, scores, Xc = pca_on_connectivity_profiles(Zm, var_keep=None, random_state=random_state)
        evr = pca.explained_variance_ratio_; eigs = pca.explained_variance_; cum = np.cumsum(evr)

        if fast_mode:
            k_star = np.argmax(cum >= k_target_variance) + 1 if np.any(cum >= k_target_variance) else len(evr)
        else:
            bs = broken_stick_expectation(len(evr))
            pa = parallel_analysis(Xc, n_perm=200, random_state=random_state)
            k_cum = np.argmax(cum >= k_target_variance) + 1 if np.any(cum >= k_target_variance) else len(evr)
            k_pa  = int(np.sum(eigs > pa[:len(eigs)]))
            k_bs  = int(np.sum(evr  > bs[:len(evr)]))
            k_star = max(1, min(k_cum, max(k_pa, k_bs)))

        # Only write figures/tables when allowed
        if write_artifacts and not fast_mode:
            plot_explained_variance(pca, f"{tag} • {phase} • PCA variance explained",
                                    prefix + "_pca_variance.png")
            plot_pca_scatter(scores, roi_labels, 1, 2,
                             f"{tag} • {phase} • PCA scatter (PC1 vs PC2)",
                             prefix + "_pca_scatter_PC1_PC2.png")
            K_HEAT = min(8, scores.shape[1])
            plot_loadings_heatmap(pca, roi_labels, K_HEAT,
                                  f"{tag} • {phase} • PCA loadings (first {K_HEAT} PCs)",
                                  prefix + f"_pca_loadings_heatmap_first{K_HEAT}.png")
            for pc in (1,2,3,4):
                if pc <= scores.shape[1]:
                    plot_top_loading_bars(pca, roi_labels, pc, 12,
                                          f"{tag} • {phase} • Top 12 |loadings| (PC{pc})",
                                          prefix + f"_pca_loadings_PC{pc}_top12.png")
            # stability only in full mode
            if n_boot_pc_stability and n_boot_pc_stability > 0:
                stab = bootstrap_pc_stability(Xc, n_boot=n_boot_pc_stability,
                                              random_state=random_state, k_check=(1,2))
                pd.DataFrame([{"pc":1,"median_abs_corr":stab.get(1,np.nan)},
                              {"pc":2,"median_abs_corr":stab.get(2,np.nan)}]
                            ).to_csv(prefix + "_pca_stability.csv", index=False)
            with open(prefix + "_pca_kstar.txt","w") as f:
                f.write(f"k_star={k_star}\n")

            export_pca_tables(pca, scores, roi_labels, out_prefix=prefix)

        # Clustering routes (kept in both modes)
        for meth in linkages:
            for k in n_clusters_list:
                order, labels, Z_olo = cluster_route_A(R, roi_labels, linkage_method=meth, k=k)
                if write_artifacts:
                    plot_dendro(Z_olo, [roi_labels[i] for i in order],
                                f"{tag} • {phase} • {meth} (k={k})",
                                prefix + f"_corr-{meth}_k{k}_dendrogram.png")
                    heatmap_reordered(R, order, roi_labels,
                                      f"{tag} • {phase} • corr heatmap ({meth}, k={k})",
                                      prefix + f"_corr-{meth}_k{k}_heatmap.png")
                y = squareform(corr_distance_from_R(R), checks=False)
                coph_corr, _ = cophenet(Z_olo, y)
                wb_gap = within_between_block_score(R, labels)
                all_label_tables.append(pd.DataFrame({
                    "roi": roi_labels, "cluster": labels,
                    "scheme": f"corr-{meth}-k{k}", "condition": condition_name,
                    "group": group_name, "phase": phase
                }))
                all_cluster_metrics.append({
                    "condition": condition_name, "group": group_name, "phase": phase,
                    "scheme": f"corr-{meth}-k{k}",
                    "cophenetic_corr": coph_corr, "within_between_gap": wb_gap
                })

        # Ward/PCA fixed PCs
        for k in n_clusters_list:
            order, labels, Z_w, _, scores_k = cluster_route_B_Ward_PCA_fixed_k(
                Zm, roi_labels, k_pcs=K_PCS_FOR_WARD, k_clusters=k
            )
            if write_artifacts:
                plot_dendro(Z_w, [roi_labels[i] for i in order],
                            f"{tag} • {phase} • Ward/PCA (k={k}, PCs={K_PCS_FOR_WARD})",
                            prefix + f"_wardpca_k{k}_dendrogram.png")
                heatmap_reordered(R, order, roi_labels,
                                  f"{tag} • {phase} • corr heatmap (Ward/PCA k={k})",
                                  prefix + f"_wardpca_k{k}_heatmap.png")

            y_scores = pdist(scores_k)
            coph_corr_w, _ = cophenet(Z_w, y_scores)
            wb_gap_w = within_between_block_score(R, labels)
            all_label_tables.append(pd.DataFrame({
                "roi": roi_labels, "cluster": labels,
                "scheme": f"wardpca{K_PCS_FOR_WARD}-k{k}", "condition": condition_name,
                "group": group_name, "phase": phase
            }))
            all_cluster_metrics.append({
                "condition": condition_name, "group": group_name, "phase": phase,
                "scheme": f"wardpca{K_PCS_FOR_WARD}-k{k}",
                "cophenetic_corr": coph_corr_w, "within_between_gap": wb_gap_w
            })

    # Save only if writing artifacts
    label_df = pd.concat(all_label_tables, ignore_index=True)
    metrics_df = pd.DataFrame(all_cluster_metrics)
    if write_artifacts:
        label_df.to_csv(os.path.join(results_dir, f"{tag}_roi_clusters.csv"), index=False)
        metrics_df.to_csv(os.path.join(results_dir, f"{tag}_cluster_metrics.csv"), index=False)
        print(f"  [OK] ({tag}) wrote clusters/metrics")

    return {
        "roi_labels": roi_labels,
        "Z_mean": Z_mean,
        "R_mean": R_mean,
        "labels": label_df,
        "cluster_metrics": metrics_df,
        "trial_paths": list(trial_paths)
    }


def run_condition_grouped(condition_name: str,
                          all_trial_paths: list,
                          results_dir: str,
                          **kwargs):
    groups = partition_paths_by_genotype(all_trial_paths)
    if mp.current_process().name == "MainProcess":
        print(f"[INFO] {condition_name}: Found genotypes → {list(groups.keys())}")
    outputs = {}
    for g, paths in groups.items():
        print(f"\n[INFO] Processing {condition_name} • genotype={g} • n_trials={len(paths)}")
        outputs[g] = run_condition_for_group(condition_name, g, paths, results_dir, **kwargs)
    # also store raw partition so permutation tests can reassign
    outputs["_trial_partition"] = groups
    return outputs

# =========================
# 10) GENOTYPE COMPARISON HELPERS
# =========================

def _subset_labels_df(labels_df, phase: str, scheme: str):
    df = labels_df.query("phase == @phase and scheme == @scheme").copy()
    df = df.sort_values("roi")
    return df["roi"].to_list(), df["cluster"].to_numpy()

def _optimal_label_alignment(y_true: np.ndarray, y_pred: np.ndarray):
    from scipy.optimize import linear_sum_assignment
    L = max(int(y_true.max()), int(y_pred.max())) + 1
    C = np.zeros((L, L), dtype=int)
    for a, b in zip(y_true, y_pred):
        C[int(a), int(b)] += 1
    row_ind, col_ind = linear_sum_assignment(C.max() - C)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    y_pred_aligned = np.array([mapping.get(int(c), int(c)) for c in y_pred], dtype=int)
    overlap = C[row_ind, col_ind].sum() / len(y_true)
    return y_pred_aligned, float(overlap)

def _variation_of_information(y1, y2):
    from collections import Counter, defaultdict
    n = len(y1)
    c1 = Counter(y1); c2 = Counter(y2)
    def H(c):
        p = np.array(list(c.values()), dtype=float)/n
        p = p[p>0]
        return float(-(p*np.log(p)).sum())
    joint = defaultdict(int)
    for a,b in zip(y1,y2): joint[(int(a),int(b))] += 1
    mi = 0.0
    for (a,b), v in joint.items():
        pa = c1[a]/n; pb = c2[b]/n; pab = v/n
        mi += pab*np.log(pab/(pa*pb))
    return H(c1) + H(c2) - 2*mi

def _cophenetic_from_linkage(Z):
    n_leaves = Z.shape[0] + 1
    dummy = pdist(np.eye(n_leaves))
    coph_corr, coph_dists = cophenet(Z, dummy)
    return coph_dists

def mantel_test(coph1, coph2, n_perm=1000, method="spearman", random_state=0):
    rng = np.random.default_rng(random_state)
    obs = spearmanr(coph1, coph2).correlation if method=="spearman" else np.corrcoef(coph1, coph2)[0,1]
    n = int((1 + np.sqrt(1+8*len(coph1)))//2)
    idx_pairs = [(i,j) for i in range(n) for j in range(i+1,n)]
    p = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        remap = []
        for (i,j) in idx_pairs:
            a, b = sorted((perm[i], perm[j]))
            remap.append(a*n - a*(a+1)//2 + (b - a - 1))
        c2p = coph2[np.array(remap, dtype=int)]
        stat = spearmanr(coph1, c2p).correlation if method=="spearman" else np.corrcoef(coph1, c2p)[0,1]
        if abs(stat) >= abs(obs): p += 1
    pval = (p+1)/(n_perm+1)
    return float(obs), float(pval)

# --- Label similarity (point estimate only; p-values via permutation below)
def label_similarity_metrics(outA, outB, phase: str, scheme: str):
    roisA, labA = _subset_labels_df(outA["labels"], phase, scheme)
    roisB, labB = _subset_labels_df(outB["labels"], phase, scheme)
    assert roisA == roisB, "ROI orders differ between groups."

    ari = adjusted_rand_score(labA, labB)
    nmi = normalized_mutual_info_score(labA, labB)
    vi  = _variation_of_information(labA, labB)
    fm  = fowlkes_mallows_score(labA, labB)
    labB_aligned, overlap = _optimal_label_alignment(labA, labB)
    return {
        "ARI": float(ari),
        "NMI": float(nmi),
        "VI":  float(vi),
        "FM":  float(fm),
        "overlap_after_alignment": float(overlap),
        "labels_A": labA, "labels_B": labB, "labels_B_aligned": labB_aligned, "rois": roisA
    }

# --- Tree shape comparison (Mantel) ---
def tree_comparison_mantel(outA, outB, phase: str, scheme: str, method="spearman", n_perm=1000):
    # Rebuild Z for the scheme using each group's data
    dfA = outA["labels"].query("phase == @phase and scheme == @scheme").sort_values("roi")
    dfB = outB["labels"].query("phase == @phase and scheme == @scheme").sort_values("roi")
    rois = dfA["roi"].to_list()
    assert rois == dfB["roi"].to_list(), "ROI orders differ."

    if scheme.startswith("corr-"):
        meth = "average" if "average" in scheme else "complete"
        RA = outA["R_mean"][phase]; RB = outB["R_mean"][phase]
        idx = np.arange(len(rois))
        DA = corr_distance_from_R(RA[np.ix_(idx, idx)]); yA = squareform(DA, checks=False)
        DB = corr_distance_from_R(RB[np.ix_(idx, idx)]); yB = squareform(DB, checks=False)
        ZA = optimal_leaf_ordering(linkage(yA, method=meth), yA)
        ZB = optimal_leaf_ordering(linkage(yB, method=meth), yB)
    else:
        # 'wardpca6-k6'
        ZA = linkage(pdist(cluster_route_B_Ward_PCA_fixed_k(outA["Z_mean"][phase], rois, K_PCS_FOR_WARD, k_clusters=6)[4]), method="ward")
        ZB = linkage(pdist(cluster_route_B_Ward_PCA_fixed_k(outB["Z_mean"][phase], rois, K_PCS_FOR_WARD, k_clusters=6)[4]), method="ward")

    cA = _cophenetic_from_linkage(ZA)
    cB = _cophenetic_from_linkage(ZB)
    stat, pval = mantel_test(cA, cB, n_perm=n_perm, method=method, random_state=0)
    return {"mantel_corr": stat, "pval": pval}

# =========================
# 11) PERMUTATION TESTS (TRIAL-LEVEL) + BOOTSTRAP GAP
# =========================

def _recompute_group_outputs_from_paths(condition_name, group_paths_dict, results_dir,
                                        n_clusters_list, linkages, batch_size, n_workers, random_state):
    outs = {}
    for g, paths in group_paths_dict.items():
        outs[g] = run_condition_for_group(
            condition_name, g, paths, results_dir,
            n_clusters_list=n_clusters_list, linkages=linkages,
            k_target_variance=K_TARGET_VARIANCE,
            n_boot_pc_stability=0,          # no stability in resampling
            batch_size=batch_size, n_workers=n_workers, random_state=random_state,
            fast_mode=True,                 # cheaper k*
            write_artifacts=False           # <<< DO NOT WRITE FILES
        )
    return outs


def permutation_test_labels(condition_outputs: dict,
                            condition_name: str,
                            phase: str,
                            scheme: str,
                            g1: str, g2: str,
                            n_perm: int = 200,
                            random_state: int = 0,
                            batch_size: int = 256,
                            n_workers: int | None = None):
    """
    Shuffle genotype labels across *trial paths*; recompute mean R + clustering per perm;
    compute ARI/NMI/VI/FM vs the observed pair; return p-values (two-sided by tail symmetry).
    """
    rng = np.random.default_rng(random_state)
    partition = condition_outputs["_trial_partition"]
    # observed
    obs = label_similarity_metrics(condition_outputs[g1], condition_outputs[g2], phase, scheme)

    # stack all paths and labels
    all_paths, all_labels = [], []
    for g, paths in partition.items():
        for p in paths:
            all_paths.append(p); all_labels.append(g)
    all_paths = np.array(all_paths, dtype=object)
    all_labels = np.array(all_labels, dtype=object)

    # counts to preserve per genotype size
    uniq = list(partition.keys())
    counts = [len(partition[u]) for u in uniq]

    # helper to split by a permuted label multiset
    def _split_paths_by_labels(paths, labels, uniques, counts):
        # Sample a multiset by shuffling indices then slicing counts
        idx = rng.permutation(len(paths))
        start = 0
        groups = {}
        for u, c in zip(uniques, counts):
            sel = idx[start:start+c]; start += c
            groups[u] = list(paths[sel])
        return groups

    # permutation loop
    stats = {"ARI": [], "NMI": [], "VI": [], "FM": [], "overlap": []}
    for _ in range(n_perm):
        perm_groups = _split_paths_by_labels(all_paths, all_labels, [g1, g2], [len(partition[g1]), len(partition[g2])])
        # recompute only for g1,g2
        perm_outs = _recompute_group_outputs_from_paths(
            condition_name, perm_groups, RESULTS_DIR,
            n_clusters_list=N_CLUSTERS_LIST, linkages=LINKAGES,
            batch_size=batch_size, n_workers=n_workers, random_state=0
        )
        m = label_similarity_metrics(perm_outs[g1], perm_outs[g2], phase, scheme)
        stats["ARI"].append(m["ARI"])
        stats["NMI"].append(m["NMI"])
        stats["VI"].append(m["VI"])
        stats["FM"].append(m["FM"])
        stats["overlap"].append(m["overlap_after_alignment"])

    # p-values: for similarity metrics (higher = more similar), test low tail; for VI (distance), test high tail.
    def pval_low(emp, dist):
        dist = np.array(dist)
        return float((np.sum(dist <= emp) + 1) / (len(dist) + 1))
    def pval_high(emp, dist):
        dist = np.array(dist)
        return float((np.sum(dist >= emp) + 1) / (len(dist) + 1))

    pvals = {
        "p_ARI": pval_low(obs["ARI"], stats["ARI"]),
        "p_NMI": pval_low(obs["NMI"], stats["NMI"]),
        "p_FM":  pval_low(obs["FM"],  stats["FM"]),
        "p_overlap": pval_low(obs["overlap_after_alignment"], stats["overlap"]),
        "p_VI":  pval_high(obs["VI"], stats["VI"])  # higher VI = more different
    }
    return {"observed": obs, "null_dist": stats, "pvalues": pvals}

def bootstrap_within_between_gap(condition_outputs: dict,
                                 condition_name: str,
                                 phase: str,
                                 scheme: str,
                                 g1: str, g2: str,
                                 B: int = 500,
                                 random_state: int = 0,
                                 batch_size: int = 256,
                                 n_workers: int | None = None):
    """
    Bootstrap trials within each genotype, recompute clustering and within–between gap.
    Safe with B=0/1 and with occasional failed resamples.
    """
    rng = np.random.default_rng(random_state)
    partition = condition_outputs["_trial_partition"]

    def _one_gap(paths):
        outs = _recompute_group_outputs_from_paths(
            condition_name, {"G": paths}, RESULTS_DIR,
            n_clusters_list=N_CLUSTERS_LIST, linkages=LINKAGES,
            batch_size=batch_size, n_workers=n_workers, random_state=0
        )["G"]
        try:
            rois, labels = _subset_labels_df(outs["labels"], phase, scheme)
            if len(labels) == 0:
                return np.nan
            R = outs["R_mean"][phase]
            return within_between_block_score(R, labels)
        except Exception:
            return np.nan

    paths_g1 = partition[g1]; paths_g2 = partition[g2]
    n1, n2 = len(paths_g1), len(paths_g2)

    gaps_g1, gaps_g2 = [], []
    for _ in range(max(B, 0)):
        boot1 = list(np.random.choice(paths_g1, size=n1, replace=True))
        boot2 = list(np.random.choice(paths_g2, size=n2, replace=True))
        gaps_g1.append(_one_gap(boot1))
        gaps_g2.append(_one_gap(boot2))

    gaps_g1 = np.array([x for x in gaps_g1 if np.isfinite(x)], dtype=float)
    gaps_g2 = np.array([x for x in gaps_g2 if np.isfinite(x)], dtype=float)

    # If no successful samples, fall back to point estimates and NaN CIs.
    def point_gap(genotype):
        outs = condition_outputs[genotype]
        rois, labels = _subset_labels_df(outs["labels"], phase, scheme)
        R = outs["R_mean"][phase]
        return within_between_block_score(R, labels)

    if gaps_g1.size == 0 or gaps_g2.size == 0:
        g1_gap = point_gap(g1); g2_gap = point_gap(g2)
        diff = np.array([g1_gap - g2_gap])
        def _nan_ci(): return (np.nan, np.nan)
        print(f"[WARN] Bootstrap produced {gaps_g1.size}/{gaps_g2.size} samples (phase={phase}, scheme={scheme}). "
              "Returning NaN CIs; using point estimates.")
        return {
            "g1_gap_CI95": _nan_ci(), "g2_gap_CI95": _nan_ci(),
            "diff_gap_CI95": _nan_ci(),
            "diff_median": float(np.median(diff)),
            "p_two_sided": np.nan,
            "samples": {"g1": gaps_g1, "g2": gaps_g2, "diff": diff}
        }

    diff = gaps_g1 - gaps_g2

    def ci(a, alpha=0.05):
        if a.size < 2:
            v = float(np.median(a))
            return (v, v)
        lo, hi = np.quantile(a, [alpha/2, 1 - alpha/2])
        return float(lo), float(hi)

    ci1 = ci(gaps_g1); ci2 = ci(gaps_g2); cid = ci(diff)
    p_two_sided = float((np.sum(np.abs(diff) >= np.abs(np.median(diff))) + 1) / (diff.size + 1))

    print(f"[INFO] Bootstrap successes: g1={gaps_g1.size}, g2={gaps_g2.size}, diff={diff.size} "
          f"(phase={phase}, scheme={scheme})")

    return {
        "g1_gap_CI95": ci1, "g2_gap_CI95": ci2,
        "diff_gap_CI95": cid,
        "diff_median": float(np.median(diff)),
        "p_two_sided": p_two_sided,
        "samples": {"g1": gaps_g1, "g2": gaps_g2, "diff": diff}
    }


# =========================
# GENOTYPE DIFFERENCE PLOTS
# =========================
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

def _pca_from_Zmean(Z_mean):
    pca, scores, _ = pca_on_connectivity_profiles(Z_mean, var_keep=None, random_state=0)
    return pca, scores

def plot_pc_scatter_overlay(outA, outB, phase: str, labelA: str, labelB: str, outpath: str):
    ZmA = outA["Z_mean"][phase]; ZmB = outB["Z_mean"][phase]
    rois = outA["roi_labels"]; assert rois == outB["roi_labels"]
    pcaA, sA = _pca_from_Zmean(ZmA)
    pcaB, sB = _pca_from_Zmean(ZmB)

    fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(sA[:,0], sA[:,1], s=32, label=labelA, alpha=0.8)
    ax.scatter(sB[:,0], sB[:,1], s=32, label=labelB, alpha=0.8, marker='^')

    # convex hulls (if enough points)
    for pts, col, lab in [(sA, None, labelA), (sB, None, labelB)]:
        if pts.shape[0] >= 3:
            hull = ConvexHull(pts[:, :2])
            poly = Polygon(pts[hull.vertices, :2], closed=True, fill=False, lw=1.2)
            ax.add_patch(poly)

    for i, r in enumerate(rois):
        ax.text(sA[i,0], sA[i,1], r, fontsize=7, alpha=0.5)  # annotate by A position

    ax.axhline(0, ls=':', lw=0.8, color='k'); ax.axvline(0, ls=':', lw=0.8, color='k')
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"PC1–PC2 overlay • {phase}")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(outpath, dpi=220); plt.close(fig)

def plot_R_difference_heatmap(outA, outB, phase: str, title: str, outpath: str, vlim=0.4):
    rois = outA["roi_labels"]; assert rois == outB["roi_labels"]
    RA = outA["R_mean"][phase]; RB = outB["R_mean"][phase]
    dR = RA - RB
    plt.figure(figsize=(6.8,6.0))
    sns.heatmap(dR, cmap="coolwarm", center=0, vmin=-vlim, vmax=vlim,
                xticklabels=rois, yticklabels=rois, square=True)
    plt.title(title); plt.tight_layout(); plt.savefig(outpath, dpi=220); plt.close()

def plot_cluster_contingency(outA, outB, phase: str, scheme: str, outpath: str):
    roisA, labA = _subset_labels_df(outA["labels"], phase, scheme)
    roisB, labB = _subset_labels_df(outB["labels"], phase, scheme)
    assert roisA == roisB
    K = int(max(labA.max(), labB.max()))
    M = np.zeros((K, K), dtype=int)
    for a,b in zip(labA, labB): M[int(a)-1, int(b)-1] += 1
    plt.figure(figsize=(4.5,4.0))
    sns.heatmap(M, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=[f"B{j+1}" for j in range(K)],
                yticklabels=[f"A{i+1}" for i in range(K)])
    plt.xlabel("Genotype B clusters"); plt.ylabel("Genotype A clusters")
    plt.title(f"Cluster mapping • {phase} • {scheme}")
    plt.tight_layout(); plt.savefig(outpath, dpi=220); plt.close()

def plot_dendrogram_side_by_side(outA, outB, phase: str, scheme: str, labelA: str, labelB: str, outpath: str):
    roisA, _ = _subset_labels_df(outA["labels"], phase, scheme)
    roisB, _ = _subset_labels_df(outB["labels"], phase, scheme)
    assert roisA == roisB
    rois = roisA

    if scheme.startswith("corr-"):
        meth = "average" if "average" in scheme else "complete"
        for outs, lab, col in [(outA, labelA, 1), (outB, labelB, 2)]:
            R = outs["R_mean"][phase]; D = corr_distance_from_R(R); y = squareform(D, checks=False)
            Z = optimal_leaf_ordering(linkage(y, method=meth), y)
            plt.figure(figsize=(8.5,3.6))
            dendrogram(Z, labels=rois, leaf_rotation=90)
            plt.title(f"{lab} • {phase} • {scheme}")
            plt.tight_layout(); plt.savefig(outpath.replace(".png", f"_{lab}.png"), dpi=220); plt.close()
    else:
        for outs, lab in [(outA, labelA), (outB, labelB)]:
            Zm = outs["Z_mean"][phase]
            _, _, Z, _, _ = cluster_route_B_Ward_PCA_fixed_k(Zm, rois, k_pcs=K_PCS_FOR_WARD, k_clusters=6)
            plt.figure(figsize=(8.5,3.6))
            dendrogram(Z, labels=rois, leaf_rotation=90)
            plt.title(f"{lab} • {phase} • {scheme}")
            plt.tight_layout(); plt.savefig(outpath.replace(".png", f"_{lab}.png"), dpi=220); plt.close()

def plot_gap_bars_with_ci(boot_res: dict, g1: str, g2: str, outpath: str):
    # expects output from bootstrap_within_between_gap(...)
    ci1_lo, ci1_hi = boot_res["g1_gap_CI95"]
    ci2_lo, ci2_hi = boot_res["g2_gap_CI95"]
    med1 = float(np.median(boot_res["samples"]["g1"]))
    med2 = float(np.median(boot_res["samples"]["g2"]))

    fig, ax = plt.subplots(figsize=(4.6,4.2))
    xs = np.arange(2)
    meds = [med1, med2]
    yerr = [[med1 - ci1_lo, med2 - ci2_lo],
            [ci1_hi - med1, ci2_hi - med2]]
    ax.bar(xs, meds, width=0.6)
    ax.errorbar(xs, meds, yerr=yerr, fmt='none', capsize=6, lw=1.2, color='k')
    ax.set_xticks(xs); ax.set_xticklabels([g1, g2])
    ax.set_ylabel("Within − Between corr (gap)")
    ax.set_title("Modularity strength with 95% CI")
    fig.tight_layout(); fig.savefig(outpath, dpi=220); plt.close(fig)


# =========================
# 12) MAIN PIPELINE
# =========================

def main_pipeline(base_dir: str,
                  results_dir: str,
                  n_clusters_list=N_CLUSTERS_LIST,
                  linkages=LINKAGES,
                  k_target_variance=K_TARGET_VARIANCE,
                  n_boot_pc_stability=200,
                  batch_size=256,
                  n_workers=None,
                  random_state=0):
    """
    Expects subfolders under base_dir:
      base_dir/
        ├── stim_day_3/
        └── intrinsic_day_2/
    Groups runs by genotype within each condition.
    """
    stim_dir   = os.path.join(base_dir, "stim_day_3")
    intrin_dir = os.path.join(base_dir, "intrinsic_day_2")
    print(f"[INFO] Looking for trials in:\n  {stim_dir}\n  {intrin_dir}")

    stim_paths   = list_trial_paths(stim_dir,   recursive=True)
    intrin_paths = list_trial_paths(intrin_dir, recursive=True)

    out_stim = run_condition_grouped(
        'stim_day3', stim_paths, results_dir,
        n_clusters_list=n_clusters_list,
        linkages=linkages,
        k_target_variance=k_target_variance,
        n_boot_pc_stability=n_boot_pc_stability,
        batch_size=batch_size,
        n_workers=n_workers,
        random_state=random_state
    )

    out_intrin = run_condition_grouped(
        'intrinsic_day2', intrin_paths, results_dir,
        n_clusters_list=n_clusters_list,
        linkages=linkages,
        k_target_variance=k_target_variance,
        n_boot_pc_stability=n_boot_pc_stability,
        batch_size=batch_size,
        n_workers=n_workers,
        random_state=random_state
    )

    return out_stim, out_intrin

# =========================
# METRIC HELPERS (gap, Q, silhouette, hemisphere-bias)
# =========================
from sklearn.metrics import silhouette_score

def _hemi_sides(roi_labels):
    L = [i for i,r in enumerate(roi_labels) if r.startswith("L-")]
    R = [i for i,r in enumerate(roi_labels) if r.startswith("R-")]
    return L, R

def modularity_Q_weighted(R: np.ndarray, labels: np.ndarray, zero_negative: bool = True) -> float:
    """
    Newman–Girvan modularity for an undirected weighted graph.
    Uses strengths; optionally zeroes negative weights (recommended for correlations).
    """
    A = np.array(R, dtype=float, copy=True)
    np.fill_diagonal(A, 0.0)
    if zero_negative:
        A[A < 0.0] = 0.0
    k = A.sum(axis=1)                    # strengths
    m = k.sum() / 2.0
    if m <= 1e-12:
        return float('nan')
    P = np.outer(k, k) / (2.0 * m)       # configuration null
    same = (labels[:, None] == labels[None, :]).astype(float)
    Q = ((A - P) * same).sum() / (2.0 * m)
    return float(Q)

def hemisphere_bias_index(R: np.ndarray, roi_labels: list[str]) -> float:
    """
    Mean within-hemisphere correlation minus mean cross-hemisphere correlation.
    Positive ⇒ more hemispheric modularity.
    """
    A = np.array(R, dtype=float, copy=True)
    np.fill_diagonal(A, np.nan)
    L, Ridx = _hemi_sides(roi_labels)
    # within: LL and RR
    LL = A[np.ix_(L, L)]; RR = A[np.ix_(Ridx, Ridx)]
    within_vals = np.concatenate([LL[np.triu_indices(len(L),1)], RR[np.triu_indices(len(Ridx),1)]])
    # between: LR
    LR = A[np.ix_(L, Ridx)].ravel()
    w = np.nanmean(within_vals) if within_vals.size else np.nan
    b = np.nanmean(LR) if LR.size else np.nan
    return float(w - b)

def silhouette_mean_from_R(R: np.ndarray, labels: np.ndarray) -> float:
    """
    Silhouette using correlation distance D = sqrt(2*(1-R)).
    Returns NaN if fewer than 2 clusters with >=2 members.
    """
    # Check cluster sizes
    _, counts = np.unique(labels, return_counts=True)
    if np.sum(counts >= 2) < 2:
        return float('nan')
    D = corr_distance_from_R(R)
    try:
        return float(silhouette_score(D, labels, metric="precomputed"))
    except Exception:
        return float('nan')

def compute_metrics_for_condition(condition_outputs: dict,
                                  condition_name: str,
                                  phases: list[str],
                                  scheme: str) -> pd.DataFrame:
    """
    condition_outputs: dict returned by run_condition_grouped(...), e.g., out_stim
    Returns a tidy DataFrame with one row per (condition, phase, genotype).
    """
    rows = []
    for g in sorted(k for k in condition_outputs.keys() if k != "_trial_partition"):
        outs = condition_outputs[g]
        roi_labels = outs["roi_labels"]
        for phase in phases:
            # cluster labels for the chosen scheme
            _, labels = _subset_labels_df(outs["labels"], phase, scheme)
            # connectivity (R)
            R = outs["R_mean"][phase]
            # metrics
            gap = within_between_block_score(R, labels)
            Q = modularity_Q_weighted(R, labels, zero_negative=True)
            sil = silhouette_mean_from_R(R, labels)
            hemi = hemisphere_bias_index(R, roi_labels)
            rows.append({
                "condition": condition_name, "phase": phase, "genotype": g,
                "within_between_gap": gap, "Q_weighted": Q,
                "silhouette_mean": sil, "hemisphere_bias": hemi
            })
    return pd.DataFrame(rows)

def compute_all_metrics(out_stim: dict, out_intrin: dict, scheme: str = "wardpca6-k6") -> pd.DataFrame:
    """Convenience: run both conditions with the same scheme."""
    phases = list(PHASES.keys())  # ['pre','during','post']
    df1 = compute_metrics_for_condition(out_stim,   "stim_day3",   phases, scheme)
    df2 = compute_metrics_for_condition(out_intrin, "intrinsic_day2", phases, scheme)
    return pd.concat([df1, df2], ignore_index=True)

# =========================
# 2×3 SUMMARY TABLES + BAR PLOTS
# =========================
def pivot_2x3(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Builds a 2×3 table with columns MultiIndex (condition, phase) and rows = genotype.
    Order: columns [('stim_day3','pre'), ('stim_day3','during'), ('stim_day3','post'),
                    ('intrinsic_day2','pre'), ('intrinsic_day2','during'), ('intrinsic_day2','post')]
    """
    # enforce consistent column order
    col_order = [("stim_day3","pre"), ("stim_day3","during"), ("stim_day3","post"),
                 ("intrinsic_day2","pre"), ("intrinsic_day2","during"), ("intrinsic_day2","post")]
    pv = df.pivot_table(index="genotype", columns=["condition","phase"], values=metric, aggfunc="mean")
    # reindex columns to the desired order if present
    cols_present = [c for c in col_order if c in pv.columns]
    pv = pv.reindex(columns=cols_present)
    return pv

def plot_metric_bars(df: pd.DataFrame, metric: str, outdir: str):
    """
    Makes a genotype×phase bar plot separately for each condition; saves two PNGs.
    """
    os.makedirs(outdir, exist_ok=True)
    for cond in df["condition"].unique():
        d = df[df["condition"] == cond].copy()
        # preserve phase order
        d["phase"] = pd.Categorical(d["phase"], categories=list(PHASES.keys()), ordered=True)
        d = d.sort_values(["phase","genotype"])
        plt.figure(figsize=(6.2, 3.6))
        ax = plt.gca()
        # grouped bars: x = phase, hue = genotype
        phases = d["phase"].cat.categories
        genos = sorted(d["genotype"].unique())
        x = np.arange(len(phases))
        w = 0.35
        for i,g in enumerate(genos):
            y = [d[(d["phase"]==p) & (d["genotype"]==g)][metric].mean() for p in phases]
            ax.bar(x + (i-0.5)*w, y, width=w, label=g)
        ax.set_xticks(x); ax.set_xticklabels(phases)
        ax.set_ylabel(metric.replace("_"," "))
        ax.set_title(f"{cond} • {metric}")
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{cond}_{metric}.png"), dpi=220)
        plt.close()

def save_metric_tables_and_plots(metrics_df: pd.DataFrame, outdir: str):
    """
    Writes CSV 2×3 tables for each metric and renders per-condition bar plots.
    """
    os.makedirs(outdir, exist_ok=True)
    metrics = ["within_between_gap", "Q_weighted", "silhouette_mean", "hemisphere_bias"]
    for m in metrics:
        # 2×3 table
        pv = pivot_2x3(metrics_df, m)
        pv.to_csv(os.path.join(outdir, f"summary_2x3_{m}.csv"))
        # bars
        plot_metric_bars(metrics_df, m, outdir)


# =========================
# 13) ENTRY POINT  — ordered outputs first, heavy perms later
# =========================
if __name__ == "__main__":
    BASE_DIR = "/Users/ep/Desktop/Spectrograms"   # adjust if needed

    # ---- Tunables (safe defaults; bump later if desired) ----
    N_CLUSTERS_LIST      = (6,)          # fix to k=6 for comparability
    LINKAGES             = ("average",)  # keep one to reduce runtime (you still have Ward/PCA route)
    K_PCS_FOR_WARD       = 6
    K_TARGET_VARIANCE    = 0.80
    N_BOOT_PC_STABILITY  = 200
    BATCH_SIZE           = 256
    N_WORKERS            = None
    RANDOM_STATE         = 0

    # Bootstrap size for the gap CI plot (moderately heavy; make smaller when iterating)
    B_BOOT               = 50

    # Heavy resampling flags (leave OFF for speed; turn on once visuals/stats look good)
    RUN_PERMUTATIONS     = False   # trial-level label similarity perms (very heavy)
    PERM_N               = 200
    PERM_BATCH_SIZE      = 128

    RUN_MANTEL_PERMS     = True    # Mantel test on tree shape (moderate)
    MANTEL_N             = 1000

    # Clustering scheme for comparison figures
    SCHEME               = "wardpca6-k6"   # or "corr-average-k6"

    # ---- Run the core pipeline once ----
    out_stim, out_intrin = main_pipeline(
        BASE_DIR,
        RESULTS_DIR,
        n_clusters_list=N_CLUSTERS_LIST,
        linkages=LINKAGES,
        k_target_variance=K_TARGET_VARIANCE,
        n_boot_pc_stability=N_BOOT_PC_STABILITY,
        batch_size=BATCH_SIZE,
        n_workers=N_WORKERS,
        random_state=RANDOM_STATE
    )

    # ---- Helper: pick canonical genotype keys robustly (case-insensitive) ----
    def _pick_key(d, target):
        for k in d.keys():
            if k == "_trial_partition": 
                continue
            if str(k).lower() == target.lower():
                return k
        return None

    def _pick_two_genotypes(d):
        # Prefer WT vs SETD1A/Setd1a if present; else first two non-partition keys
        g_wt  = _pick_key(d, "WT")
        g_set = None
        for cand in ("SETD1A", "Setd1a", "KO", "HET"):
            g_set = _pick_key(d, cand)
            if g_set: break
        keys = [k for k in d.keys() if k != "_trial_partition"]
        if g_wt and g_set: 
            return g_wt, g_set
        if len(keys) >= 2:
            return keys[0], keys[1]
        raise RuntimeError("Could not identify two genotype groups.")

    # Choose consistent g1/g2 using the stim outputs (same keys exist in intrinsic)
    g1, g2 = _pick_two_genotypes(out_stim)
    print(f"[INFO] Using genotypes: g1={g1} vs g2={g2}")

    # ---- Generate figures + core stats in desired order, for each condition × phase ----
    for cond_name, cond_out in [("stim_day3", out_stim), ("intrinsic_day2", out_intrin)]:
        print(f"\n[INFO] === {cond_name} ===")
        for phase in PHASES.keys():  # 'pre', 'during', 'post'
            print(f"[INFO] Phase: {phase}")

            # 1) PC1–PC2 overlay
            plot_pc_scatter_overlay(
                cond_out[g1], cond_out[g2], phase, g1, g2,
                os.path.join(RESULTS_DIR, f"{cond_name}_{phase}_PC12_overlay_{g1}_vs_{g2}.png")
            )

            # 2) ΔR heatmap (R_g1 − R_g2)
            plot_R_difference_heatmap(
                cond_out[g1], cond_out[g2], phase,
                f"ΔR = R_{g1} − R_{g2} • {cond_name} • {phase}",
                os.path.join(RESULTS_DIR, f"{cond_name}_{phase}_dR_{g1}_minus_{g2}.png"),
                vlim=0.4
            )

            # 3) Cluster mapping (contingency) at chosen scheme
            plot_cluster_contingency(
                cond_out[g1], cond_out[g2], phase, SCHEME,
                os.path.join(RESULTS_DIR, f"{cond_name}_{phase}_{SCHEME}_cluster_contingency_{g1}_vs_{g2}.png")
            )

            # 4) Side-by-side dendrograms (saves two files: _<g1>.png and _<g2>.png)
            plot_dendrogram_side_by_side(
                cond_out[g1], cond_out[g2], phase, SCHEME, g1, g2,
                os.path.join(RESULTS_DIR, f"{cond_name}_{phase}_{SCHEME}_dendrograms_{g1}_vs_{g2}.png")
            )

            # 5) Gap CI bars via bootstrap
            boot_res = bootstrap_within_between_gap(
                cond_out, cond_name, phase, SCHEME, g1, g2,
                B=B_BOOT, random_state=RANDOM_STATE,
                batch_size=PERM_BATCH_SIZE, n_workers=N_WORKERS
            )
            plot_gap_bars_with_ci(
                boot_res, g1, g2,
                os.path.join(RESULTS_DIR, f"{cond_name}_{phase}_{SCHEME}_gap_CIs_{g1}_vs_{g2}.png")
            )
            # Save a tiny text summary for quick reference
            with open(os.path.join(RESULTS_DIR, f"{cond_name}_{phase}_{SCHEME}_gap_summary_{g1}_vs_{g2}.txt"), "w") as f:
                f.write(
                    f"g1={g1} CI95: {boot_res['g1_gap_CI95']}\n"
                    f"g2={g2} CI95: {boot_res['g2_gap_CI95']}\n"
                    f"diff median: {boot_res['diff_median']}\n"
                    f"diff CI95: {boot_res['diff_gap_CI95']}\n"
                    f"p_two_sided: {boot_res['p_two_sided']}\n"
                )

            # ---- Associated stats (observed label similarity + Mantel) ----
            obs = label_similarity_metrics(cond_out[g1], cond_out[g2], phase, SCHEME)
            pd.DataFrame([obs]).drop(columns=["labels_A","labels_B","labels_B_aligned","rois"]).to_csv(
                os.path.join(RESULTS_DIR, f"{cond_name}_{phase}_{SCHEME}_label_metrics_{g1}_vs_{g2}.csv"),
                index=False
            )

            mantel = tree_comparison_mantel(
                cond_out[g1], cond_out[g2], phase, SCHEME,
                method="spearman", n_perm=(MANTEL_N if RUN_MANTEL_PERMS else 0)
            )
            pd.DataFrame([mantel]).to_csv(
                os.path.join(RESULTS_DIR, f"{cond_name}_{phase}_{SCHEME}_mantel_{g1}_vs_{g2}.csv"),
                index=False
            )

    print("\n[OK] Phase-by-phase overlays, ΔR, contingencies, dendrograms, gap-CIs, and core stats are saved.")

    # ---- (Optional) Heavy: trial-level permutation tests for label similarity ----
    if RUN_PERMUTATIONS:
        for cond_name, cond_out in [("stim_day3", out_stim), ("intrinsic_day2", out_intrin)]:
            for phase in PHASES.keys():
                perm = permutation_test_labels(
                    cond_out, cond_name, phase, SCHEME, g1, g2,
                    n_perm=PERM_N, random_state=RANDOM_STATE,
                    batch_size=PERM_BATCH_SIZE, n_workers=N_WORKERS
                )
                pd.DataFrame([perm["observed"]]).to_csv(
                    os.path.join(RESULTS_DIR, f"{cond_name}_{phase}_{SCHEME}_perm_label_metrics_{g1}_vs_{g2}.csv"),
                    index=False
                )
                pd.DataFrame([perm["pvalues"]]).to_csv(
                    os.path.join(RESULTS_DIR, f"{cond_name}_{phase}_{SCHEME}_perm_label_pvals_{g1}_vs_{g2}.csv"),
                    index=False
                )
        print("[OK] Permutation tests complete.")
    else:
        print("[INFO] Skipped heavy label-permutation tests (RUN_PERMUTATIONS=False).")
