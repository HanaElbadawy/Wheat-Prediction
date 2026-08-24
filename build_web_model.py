"""
Compress deploy_model.npz into a form small enough to deploy.

    python build_web_model.py

Prediction only ever needs alpha @ A_train, never A_train itself:

    pred = mu + w_A/den_A * (alpha @ A_train) @ A_new
                + w_D/den_D * (alpha @ U_train) @ U_new

Collapsing those products turns two (1725 x 8258) float32 matrices — about
114 MB in RAM — into two 8,258-element vectors, roughly 0.13 MB. The arithmetic
is identical; this is an associativity rearrangement, not an approximation.
The script asserts that below.
"""

from pathlib import Path
import numpy as np

SRC = Path(__file__).parent / "deploy_model.npz"
DST = Path(__file__).parent / "web_model.npz"


def main():
    if not SRC.exists():
        raise SystemExit(f"Missing {SRC}. Run export_deploy_model.py first.")

    d = np.load(SRC, allow_pickle=True)
    alpha = d["alpha"].astype(np.float64)
    A, U = d["A_train"].astype(np.float64), d["U_train"].astype(np.float64)
    w_A, w_D = float(d["w_A"]), float(d["w_D"])
    den_A, den_D = float(d["den_A"]), float(d["den_D"])

    beta_a = (w_A / den_A) * (alpha @ A)      # (n_markers,)
    beta_d = (w_D / den_D) * (alpha @ U)

    # Verify against the full-matrix path on a few real genotypes.
    names = d["geno_names"]
    dosages = d["geno_markers"].astype(np.float64)
    palle, d11, d12, d22 = d["Palle"], d["d11"], d["d12"], d["d22"]
    mu = float(d["mu"])

    def encode(dose):
        x = dose - 1.0
        return (x - (2 * palle - 1),
                np.where(x <= -0.5, d11, np.where(x >= 0.5, d22, d12)))

    worst = 0.0
    rng = np.random.default_rng(0)
    for _ in range(50):
        i, j = rng.integers(0, len(names), 2)
        a_new, u_new = encode((dosages[i] + dosages[j]) / 2.0)
        full = mu + (w_A * (A @ a_new) / den_A
                     + w_D * (U @ u_new) / den_D) @ alpha
        fast = mu + beta_a @ a_new + beta_d @ u_new
        worst = max(worst, abs(full - fast))

    print(f"max |full - compressed| over 50 crosses: {worst:.3e}")
    if worst > 1e-8:
        raise AssertionError("Compression changed the predictions")

    np.savez_compressed(
        DST,
        beta_a=beta_a, beta_d=beta_d, mu=mu,
        Palle=palle, d11=d11, d12=d12, d22=d22,
        geno_names=names, geno_markers=dosages.astype(np.float32),
        train_females=d["train_females"], train_males=d["train_males"],
        all_females=d["all_females"], all_males=d["all_males"],
        obs_female=d["obs_female"], obs_male=d["obs_male"],
        obs_yield=d["obs_yield"],
    )
    before = SRC.stat().st_size / 1e6
    after = DST.stat().st_size / 1e6
    print(f"{SRC.name}: {before:.1f} MB  ->  {DST.name}: {after:.1f} MB")
    print("PASS: predictions unchanged.")


if __name__ == "__main__":
    main()
