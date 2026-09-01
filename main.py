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
# Plant health / crop-vigor scanner — SegFormer, 9-band multispectral UAV
# ----------------------------------------------------------------------
#
# IMPORTANT: this is not a leaf-photo disease classifier. The delivered
# model is a per-pixel segmentation model trained on 9-band multispectral
# drone imagery (Blue/Green/Red/RedEdge/NIR + NDVI/NDRE/CI_RedEdge/GNDVI),
# predicting crop vigor (Low/Medium/High) per pixel from a 224x224 patch.
# It cannot classify an ordinary phone photo — see plant_health_model.py
# for why, and for the transformers-version pin this checkpoint requires.

import base64
import io

try:
    import plant_health_model as phm
    _SCANNER_IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001 — torch/transformers not installed on this deploy
    phm = None
    _SCANNER_IMPORT_ERROR = str(exc)


@app.get("/api/scanner/status", tags=["scanner"])
def scanner_status():
    if phm is None:
        return {
            "available": False,
            "message": "Scanner dependencies (torch/transformers) are not installed "
                       "on this deployment. The rest of the app is unaffected.",
        }
    try:
        phm.load_model()
        available = True
        message = "Model connected and ready."
    except Exception as exc:  # noqa: BLE001 — surface the real reason
        available = False
        message = f"Model failed to load: {exc}"
    return {
        "available": available,
        "message": message,
        "input_requirements": {
            "format": "9-band multispectral GeoTIFF (UAV capture)",
            "bands": phm.BAND_NAMES,
            "patch_size": phm.PATCH_SIZE,
            "not_supported": "Ordinary RGB photos (phone/webcam) cannot be scored — "
                              "they don't contain the RedEdge/NIR bands or the "
                              "pre-computed vegetation indices the model was trained on.",
        },
        "classes": phm.CLASS_NAMES,
        "reported_test_metrics": phm.REPORTED_METRICS,
        "sample_patches_available": phm.list_samples(),
    }


@app.get("/api/scanner/samples", tags=["scanner"])
def scanner_samples():
    """Bundled demo patches, since most visitors won't have their own UAV captures."""
    if phm is None:
        raise HTTPException(503, "Scanner is not available on this deployment.")
    return {"samples": phm.list_samples()}


def _run_scanner_bytes(data: bytes) -> dict:
    feature = phm.read_multiband_tif(data)
    result = phm.predict(feature)
    rgb = phm.colourise(result["pred_map"])

    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "dominant_class": result["dominant_class"],
        "class_pixel_pct": result["class_pixel_pct"],
        "mean_confidence": result["mean_confidence"],
        "map_png_base64": png_b64,
        "note": "Per-pixel crop-vigor map (Low/Medium/High), not a disease diagnosis. "
                f"Reported test-set accuracy: mIoU={phm.REPORTED_METRICS['mIoU']}, "
                f"pixel accuracy={phm.REPORTED_METRICS['pixel_accuracy']}.",
    }


@app.post("/api/scanner/predict", tags=["scanner"])
async def scanner_predict(file: UploadFile = File(...)):
    """Score an uploaded 9-band multispectral GeoTIFF patch."""
    if phm is None:
        raise HTTPException(503, "Scanner is not available on this deployment.")
    data = await file.read()
    if len(data) > 25_000_000:
        raise HTTPException(413, "File exceeds the 25 MB limit.")
    try:
        return _run_scanner_bytes(data)
    except ValueError as exc:
        raise HTTPException(415, str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc))
    except RuntimeError as exc:
        raise HTTPException(500, str(exc))


@app.post("/api/scanner/predict-sample/{name}", tags=["scanner"])
async def scanner_predict_sample(name: str):
    """Score one of the bundled demo patches by filename."""
    if phm is None:
        raise HTTPException(503, "Scanner is not available on this deployment.")
    path = phm.SAMPLE_DIR / name
    if not path.exists() or path not in {phm.SAMPLE_DIR / n for n in phm.list_samples()}:
        raise HTTPException(404, f"No sample named {name!r}.")
    try:
        return _run_scanner_bytes(path.read_bytes())
    except RuntimeError as exc:
        raise HTTPException(500, str(exc))


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
