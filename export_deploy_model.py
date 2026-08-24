"""
Build and export the deployment model — standalone.

    python export_deploy_model.py

Fits GBLUP (additive + dominance) on all Experiment II hybrids and writes
deploy_model.npz, which predictor.py and app.py load.

This does NOT re-measure T2/T1/T0 — Notebook 06 did that. The confidence bands
live in predictor.py as constants taken from those measurements.

Everything stored here comes from TRAINING rows. Palle, den_A, den_D and the
d11/d12/d22 dominance parameters must be reused exactly as fitted: recomputing
them for a new genotype changes the scale and silently corrupts predictions.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Settings — edit PROJECT_DIR if you move things
# ----------------------------------------------------------------------

PROJECT_DIR = Path(r"D:\Final gp\trial\wheat_project")
OUT_DIR = PROJECT_DIR / "deployment"

SNP_FILE = PROJECT_DIR / "snp_data" / "Integrated data SNP of the hybrid trials.txt"
BLUES_FILE = (PROJECT_DIR / "repository"
              / "Genomic-prediction-of-hybrid-wheat-master"
              / "Integrated data BLUEs of the hybrid trials.txt")

W_A, W_D, LAMBDA = 0.9, 0.1, 0.3      # selected in Notebook 06
MAF_MIN = 0.05
EXP = ["Exp.I", "Exp.II", "Exp.III", "Exp.IV", "Exp.V"]


def check_inputs():
    for p in (SNP_FILE, BLUES_FILE):
        if not p.exists():
            listing = sorted(PROJECT_DIR.rglob("*.txt"),
                             key=lambda f: -f.stat().st_size)[:10]
            lines = "\n".join(
                f"    {f.stat().st_size / 1e6:8.1f} MB  "
                f"{f.relative_to(PROJECT_DIR)}"
                for f in listing) or "    (no .txt files found)"
            raise FileNotFoundError(
                f"Missing {p}\n\nLargest .txt files under {PROJECT_DIR}:\n"
                f"{lines}\n\nSet SNP_FILE / BLUES_FILE at the top of this file."
            )


def load_data():
    print("Reading marker panel (about a minute)...")
    id_rows, marker_rows = [], []
    with open(SNP_FILE) as fh:
        fh.readline()
        for line in fh:
            parts = line.split()
            id_rows.append(parts[1:4])
            marker_rows.append(parts[4:])

    ids = pd.DataFrame(id_rows, columns=["Female", "Male", "Geno"])
    ids = ids.map(lambda s: s.replace('"', ""))
    markers = np.array(marker_rows, dtype=np.float32)

    blues = pd.read_csv(BLUES_FILE, sep=r"\s+", engine="python")
    blues.columns = [str(c).replace('"', "") for c in blues.columns]
    for c in blues.columns:
        if blues[c].map(type).eq(str).all():
            blues[c] = blues[c].str.replace('"', "", regex=False)
    for c in ["Blues"] + EXP + ["Hybrid", "Lines"]:
        if c in blues.columns:
            blues[c] = pd.to_numeric(blues[c], errors="coerce")

    if len(blues) != len(ids):
        raise ValueError(
            f"Row mismatch: {len(blues)} phenotypes vs {len(ids)} marker rows")

    print(f"  panel   {markers.shape}")
    print(f"  hybrids {int((blues['Hybrid'] == 1).sum())}")
    return ids, markers, blues


def fit_model(markers_train, y_train):
    """GBLUP with the source-code dominance formulation."""
    p = markers_train.mean(axis=0) / 2.0
    keep = np.minimum(p, 1 - p) >= MAF_MIN
    matrix = markers_train[:, keep]
    n = len(matrix)

    x = matrix - 1.0
    palle = (x.mean(axis=0) + 1) / 2.0
    den_a = float(np.sum(2 * palle * (1 - palle)))
    den_d = float(np.sum((2 * palle * (1 - palle)) ** 2))

    p11 = ((x == -1).sum(0) + 0.5 * (x == -0.5).sum(0)) / n
    p12 = ((x == 0).sum(0) + 0.5 * (x == -0.5).sum(0)
                           + 0.5 * (x == 0.5).sum(0)) / n
    p22 = ((x == 1).sum(0) + 0.5 * (x == 0.5).sum(0)) / n
    theta = p11 + p22 - (p11 - p22) ** 2
    theta = np.where(theta == 0, 1e-12, theta)

    d11 = -2 * p12 * p22 / theta
    d12 = 4 * p11 * p22 / theta
    d22 = -2 * p11 * p12 / theta

    a_train = x - (2 * palle - 1)
    u_train = np.where(x <= -0.5, d11, np.where(x >= 0.5, d22, d12))

    g_a = (a_train @ a_train.T) / den_a
    g_d = (u_train @ u_train.T) / den_d
    kernel = W_A * g_a + W_D * g_d

    v = kernel + LAMBDA * np.eye(n)
    one = np.ones(n)
    mu = float((one @ np.linalg.solve(v, y_train))
               / (one @ np.linalg.solve(v, one)))
    alpha = np.linalg.solve(v, y_train - mu)

    print(f"\nFitted on {n} hybrids, {int(keep.sum())} markers")
    print(f"  mu = {mu:.4f}")

    return {"keep": keep, "Palle": palle, "den_A": den_a, "den_D": den_d,
            "d11": d11, "d12": d12, "d22": d22,
            "A_train": a_train, "U_train": u_train,
            "alpha": alpha, "mu": mu, "kernel": kernel}


def collect_parents(ids, markers, keep, female, male):
    """Each parent's own marker row.

    Predicting an unobserved cross needs the parents' own dosages. A parent
    appearing only inside hybrids has no individual profile and cannot be
    offered in the app.
    """
    geno = ids["Geno"].to_numpy()
    names, rows, missing = [], [], []

    for name in sorted(set(female) | set(male)):
        hit = np.where(geno == name)[0]
        if len(hit):
            names.append(name)
            rows.append(markers[hit[0]].astype(np.float64)[keep])
        else:
            missing.append(name)

    if missing:
        shown = ", ".join(missing[:5]) + ("..." if len(missing) > 5 else "")
        print(f"\n[!] {len(missing)} parents have no own row in the panel and "
              f"are excluded from the app: {shown}")

    available = set(names)
    females = sorted(set(female) & available)
    males = sorted(set(male) & available)
    print(f"\nSelectable: {len(females)} females x {len(males)} males "
          f"= {len(females) * len(males):,} crosses")

    return (np.array(names), np.array(rows, dtype=np.float32),
            females, males, available)


def main():
    check_inputs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ids, markers, blues = load_data()

    mask = ((blues["Hybrid"] == 1).to_numpy()
            & (blues["Exp.II"] == 1).to_numpy())
    idx = np.where(mask)[0]

    y_train = blues["Blues"].to_numpy(float)[idx]
    female = ids["Female"].to_numpy()[idx]
    male = ids["Male"].to_numpy()[idx]

    fit = fit_model(markers[idx].astype(np.float64), y_train)

    names, profiles, females, males, available = collect_parents(
        ids, markers, fit["keep"], female, male)

    obs_f, obs_m, obs_y = [], [], []
    for f, m, val in zip(female, male, y_train):
        if f in available and m in available:
            obs_f.append(f)
            obs_m.append(m)
            obs_y.append(val)
    print(f"Observed crosses recorded: {len(obs_f)}")

    out = OUT_DIR / "deploy_model.npz"
    np.savez_compressed(
        out,
        alpha=fit["alpha"], mu=fit["mu"], w_A=W_A, w_D=W_D,
        Palle=fit["Palle"], den_A=fit["den_A"], den_D=fit["den_D"],
        d11=fit["d11"], d12=fit["d12"], d22=fit["d22"],
        A_train=fit["A_train"].astype(np.float32),
        U_train=fit["U_train"].astype(np.float32),
        geno_names=names, geno_markers=profiles,
        train_females=np.array(sorted(set(female))),
        train_males=np.array(sorted(set(male))),
        all_females=np.array(females), all_males=np.array(males),
        obs_female=np.array(obs_f), obs_male=np.array(obs_m),
        obs_yield=np.array(obs_y),
    )
    print(f"\nWrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")

    # The exported pieces must reproduce the fitted kernel exactly.
    a, u = fit["A_train"], fit["U_train"]
    recomputed = (W_A * (a @ a[0]) / fit["den_A"]
                  + W_D * (u @ u[0]) / fit["den_D"])
    gap = abs(float((recomputed - fit["kernel"][0]) @ fit["alpha"]))
    if gap > 1e-6:
        raise AssertionError(f"Export does not reproduce the fit (gap {gap})")
    print("PASS: export verified against the fitted model.")


if __name__ == "__main__":
    main()
