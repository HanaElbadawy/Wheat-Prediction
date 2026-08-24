"""
DeepHybrid API — genomic prediction of grain yield in hybrid wheat.

    uvicorn main:app --reload

Interactive docs at /docs, the interface at /.

Loads web_model.npz (see build_web_model.py), which holds the projected
coefficient vectors rather than the training matrices — same arithmetic,
about 0.2 MB instead of 114 MB, so it runs on a free hosting tier.
"""

from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE = Path(__file__).parent
MODEL_PATH = BASE / "web_model.npz"

# Measured on Experiment II, parent-level splits, 5 replicates (Notebook 06).
BANDS = {
    "T2": {"r": 0.727, "label": "High",
           "note": "both parents are in the training set"},
    "T1": {"r": 0.509, "label": "Moderate",
           "note": "one parent is new to the model"},
    "T0": {"r": 0.255, "label": "Low",
           "note": "both parents are new to the model"},
}
RANK_RELIABLE_ABOVE = 0.45

app = FastAPI(
    title="DeepHybrid API",
    description=(
        "Genomic prediction of grain yield for single-cross hybrid wheat, "
        "using GBLUP with additive and dominance relationship matrices.\n\n"
        "**Scope.** Predicts yield. It does not assess whether a cross is "
        "feasible — that depends on anther extrusion, flowering synchrony and "
        "plant height difference, which are not represented in the training "
        "data."
    ),
    version="1.0.0",
)


class Model:
    """Loaded once at startup and held in memory."""

    def __init__(self, path: Path):
        d = np.load(path, allow_pickle=True)
        self.beta_a = d["beta_a"]
        self.beta_d = d["beta_d"]
        self.mu = float(d["mu"])
        self.palle = d["Palle"]
        self.d11, self.d12, self.d22 = d["d11"], d["d12"], d["d22"]

        self.dosage = {str(k): v for k, v in
                       zip(d["geno_names"], d["geno_markers"])}
        self.train_females = set(map(str, d["train_females"]))
        self.train_males = set(map(str, d["train_males"]))
        self.females = sorted(map(str, d["all_females"]))
        self.males = sorted(map(str, d["all_males"]))
        self.observed = {
            (str(f), str(m)): float(v)
            for f, m, v in zip(d["obs_female"], d["obs_male"], d["obs_yield"])
        }
        self._dist = None
        self._sorted_all = None

    # ---- core ----

    def _encode(self, dose):
        x = dose - 1.0
        return (x - (2 * self.palle - 1),
                np.where(x <= -0.5, self.d11,
                         np.where(x >= 0.5, self.d22, self.d12)))

    def predict_one(self, female: str, male: str) -> float:
        dose = (self.dosage[female] + self.dosage[male]) / 2.0
        a, u = self._encode(dose)
        return float(self.mu + self.beta_a @ a + self.beta_d @ u)

    def predict_many(self, female: str, males: list) -> np.ndarray:
        """All crosses for one female at once."""
        stack = np.stack([self.dosage[m] for m in males])
        dose = (self.dosage[female] + stack) / 2.0
        a, u = self._encode(dose)
        return self.mu + a @ self.beta_a + u @ self.beta_d

    def scenario(self, female: str, male: str) -> str:
        seen = (female in self.train_females) + (male in self.train_males)
        return {2: "T2", 1: "T1", 0: "T0"}[seen]

    # ---- reference distribution ----

    def distribution(self) -> np.ndarray:
        """Every possible cross, computed once.

        The reference must span all combinations, not only the crosses that
        were grown — those were already selected by a breeder, which would
        bias the percentile upward.
        """
        if self._dist is None:
            rows = [self.predict_many(f, self.males) for f in self.females]
            flat = np.concatenate(rows)
            self._dist = np.sort(flat)
            self._sorted_all = [
                (f, m, float(v))
                for f, row in zip(self.females, rows)
                for m, v in zip(self.males, row)
            ]
            self._sorted_all.sort(key=lambda t: -t[2])
        return self._dist

    def percentile(self, value: float) -> float:
        dist = self.distribution()
        return float(np.searchsorted(dist, value) / len(dist) * 100)

    @staticmethod
    def rank_label(pct: float) -> str:
        if pct >= 90:
            return "Top 10% — high priority"
        if pct >= 75:
            return "Top 25%"
        if pct >= 50:
            return "Above median"
        if pct >= 25:
            return "Below median"
        return "Bottom 25%"


if not MODEL_PATH.exists():
    raise SystemExit(
        f"Missing {MODEL_PATH}\n\n"
        "Run export_deploy_model.py, then build_web_model.py.")

model = Model(MODEL_PATH)


# ----------------------------------------------------------------------
# schemas
# ----------------------------------------------------------------------

class CrossRequest(BaseModel):
    female: str
    male: str


class PredictionOut(BaseModel):
    female: str
    male: str
    yield_pred: float
    delta_vs_median: float
    scenario: str
    confidence: str
    r_value: float
    percentile: float | None
    rank_label: str
    observed: bool
    observed_value: float | None
    note: str


# ----------------------------------------------------------------------
# endpoints
# ----------------------------------------------------------------------

@app.get("/api/parents", tags=["catalogue"])
def parents():
    """Selectable parents and headline counts."""
    total = len(model.females) * len(model.males)
    return {
        "females": model.females,
        "males": model.males,
        "n_possible": total,
        "n_observed": len(model.observed),
        "n_untested": total - len(model.observed),
    }


@app.post("/api/predict", response_model=PredictionOut, tags=["prediction"])
def predict(req: CrossRequest):
    """Predict one cross and place it against all others."""
    if req.female not in model.dosage:
        raise HTTPException(404, f"Unknown female parent: {req.female}")
    if req.male not in model.dosage:
        raise HTTPException(404, f"Unknown male parent: {req.male}")

    value = model.predict_one(req.female, req.male)
    scenario = model.scenario(req.female, req.male)
    band = BANDS[scenario]

    obs = model.observed.get((req.female, req.male))
    median = float(np.median(model.distribution()))

    if band["r"] >= RANK_RELIABLE_ABOVE:
        pct = model.percentile(value)
        label = model.rank_label(pct)
        note = band["note"]
    else:
        pct, label = None, "Ranking not shown"
        note = (f"{band['note']}; at r = {band['r']:.2f} the ranking is not "
                "reliable enough to act on")

    if obs is not None:
        note = ("This cross was grown and measured — trust the observed value "
                "over the estimate.")

    return PredictionOut(
        female=req.female, male=req.male,
        yield_pred=round(value, 3),
        delta_vs_median=round(value - median, 3),
        scenario=scenario, confidence=band["label"], r_value=band["r"],
        percentile=round(pct, 1) if pct is not None else None,
        rank_label=label,
        observed=obs is not None,
        observed_value=round(obs, 3) if obs is not None else None,
        note=note,
    )


@app.get("/api/rank", tags=["prediction"])
def rank(limit: int = 25, female: str | None = None,
         untested_only: bool = True):
    """Highest-ranked crosses, optionally for one female parent."""
    if female and female not in model.dosage:
        raise HTTPException(404, f"Unknown female parent: {female}")

    model.distribution()
    rows = model._sorted_all
    if female:
        rows = [r for r in rows if r[0] == female]

    out = []
    for f, m, v in rows:
        grown = (f, m) in model.observed
        if untested_only and grown:
            continue
        out.append({"female": f, "male": m, "yield_pred": round(v, 3),
                    "status": "grown" if grown else "untested"})
        if len(out) >= min(limit, 200):
            break
    return {"count": len(out), "crosses": out}


@app.get("/api/distribution", tags=["prediction"])
def distribution(bins: int = 48):
    """Histogram of predicted yield across all possible crosses."""
    dist = model.distribution()
    counts, edges = np.histogram(dist, bins=bins)
    return {
        "counts": counts.tolist(),
        "centres": ((edges[:-1] + edges[1:]) / 2).round(3).tolist(),
        "median": round(float(np.median(dist)), 3),
        "n": len(dist),
    }


@app.get("/api/methodology", tags=["about"])
def methodology():
    """Measured performance and reproduction checks."""
    return {
        "model": "GBLUP additive + dominance",
        "training": "Experiment II — 1,725 hybrids, 8,258 markers after QC",
        "metric": "Pearson correlation between predicted and observed yield",
        "bands": BANDS,
        "reproduction": [
            {"quantity": "T2, mean over Exp I-III", "this_work": 0.756,
             "published": 0.73, "source": "main text"},
            {"quantity": "T0, mean over Exp I-III", "this_work": 0.254,
             "published": 0.25, "source": "main text"},
            {"quantity": "Exp III to Exp II", "this_work": 0.139,
             "published": 0.140, "source": "Table S6"},
            {"quantity": "T0 count, Exp II to III", "this_work": 1397,
             "published": 1397, "source": "Table S7"},
            {"quantity": "T1 count, Exp II to III", "this_work": 285,
             "published": 285, "source": "Table S7"},
        ],
        "not_modelled": [
            "Reproductive compatibility — depends on anther extrusion, "
            "flowering synchrony and plant height difference, none of which "
            "are in the training data, which contains only successful crosses",
            "Disease resistance, drought tolerance and other traits — grain "
            "yield is the only phenotype available",
            "Environmental covariates — climate variables take 5 unique "
            "values across the data, equivalent to a trial-identity factor",
        ],
        "source": ("Zhao, Y., Thorwarth, P., Jiang, Y., et al. (2021). "
                   "Science Advances 7(24), eabf9106."),
    }


class EconomicsRequest(BaseModel):
    yield_pred: float
    price_per_tonne: float = 220.0
    hectares: float = 1.0


@app.post("/api/economics", tags=["prediction"])
def economics(req: EconomicsRequest):
    """Convert a predicted yield into revenue at a user-supplied price.

    Deliberately not a market feed. Grain prices are a market variable with no
    connection to the genomic model, and live exchange data (MATIF, Euronext)
    is not freely available. The price is an input the user controls, which
    keeps the arithmetic transparent and gives the small yield differences
    between crosses a unit a breeder can act on.
    """
    dist = model.distribution()
    median = float(np.median(dist))
    best = float(dist[-1])

    rev = req.yield_pred * req.price_per_tonne * req.hectares
    return {
        "yield_pred": round(req.yield_pred, 3),
        "price_per_tonne": req.price_per_tonne,
        "hectares": req.hectares,
        "revenue": round(rev, 2),
        "revenue_per_ha": round(req.yield_pred * req.price_per_tonne, 2),
        "vs_median": round((req.yield_pred - median) * req.price_per_tonne
                           * req.hectares, 2),
        "vs_best": round((req.yield_pred - best) * req.price_per_tonne
                         * req.hectares, 2),
        "median_yield": round(median, 3),
        "best_yield": round(best, 3),
    }


# ----------------------------------------------------------------------
# Plant health scanner — integration point, model not yet available
# ----------------------------------------------------------------------
#
# A separate disease-classification model is in development by another team
# member. The interface and the contract are ready so that deployment is a
# single-file change when the model lands.
#
# TO CONNECT THE MODEL:
#   1. Put the trained weights next to this file, e.g. leaf_model.keras
#   2. Set SCANNER_MODEL_PATH below
#   3. Fill in _run_scanner() — decode, resize, predict, map to labels
# Nothing else changes: the endpoint, the schema and the page already work.

SCANNER_MODEL_PATH = None      # e.g. BASE / "leaf_model.keras"
_scanner = None


def _load_scanner():
    global _scanner
    if SCANNER_MODEL_PATH is None:
        return None
    if _scanner is None:
        import keras                       # imported lazily, only when used
        _scanner = keras.models.load_model(SCANNER_MODEL_PATH)
    return _scanner


def _run_scanner(image_bytes: bytes) -> dict:
    """Return {"label": str, "confidence": float, "alternatives": [...]}."""
    raise NotImplementedError("Wire the classifier here")


@app.get("/api/scanner/status", tags=["scanner"])
def scanner_status():
    """Whether the disease model is connected."""
    return {
        "available": SCANNER_MODEL_PATH is not None,
        "message": (
            "Model connected and ready."
            if SCANNER_MODEL_PATH is not None else
            "Disease classification model is in development. The interface "
            "and API contract are in place; predictions become available once "
            "the model is connected."
        ),
    }


@app.post("/api/scanner/predict", tags=["scanner"])
async def scanner_predict(file: UploadFile = File(...)):
    """Classify a leaf image. Returns 503 until the model is connected."""
    if SCANNER_MODEL_PATH is None:
        raise HTTPException(
            503,
            "Disease classification model is not yet connected. This endpoint "
            "is ready and will return predictions once it is.")

    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(415, "Upload an image file (JPG, PNG or TIFF).")

    data = await file.read()
    if len(data) > 20_000_000:
        raise HTTPException(413, "Image exceeds the 20 MB limit.")

    _load_scanner()
    return _run_scanner(data)


@app.get("/api/health", tags=["about"])
def health():
    return {"status": "ok", "parents": len(model.dosage)}


# ----------------------------------------------------------------------
# static front end — mounted last so /api/* wins
# ----------------------------------------------------------------------

STATIC = BASE / "static"
if STATIC.exists():
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
