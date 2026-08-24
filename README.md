# DeepHybrid

Genomic prediction of grain yield for single-cross hybrid wheat.
FastAPI backend, static front end, deployable on a free tier.

Screenshots: `screenshot_overview.png`, `screenshot_predictor.png`.

---

## Run it locally

```bash
pip install -r requirements.txt

python export_deploy_model.py    # fits the model -> deploy_model.npz
python build_web_model.py        # compresses it  -> web_model.npz
uvicorn main:app --reload
```

Open <http://127.0.0.1:8000>. The API docs are at `/docs`.

Only `web_model.npz` is needed at runtime — `deploy_model.npz` is an
intermediate and does not need to be deployed.

```
deephybrid/
├── main.py                  FastAPI app
├── static/                  index.html, styles.css, app.js
├── export_deploy_model.py   fits GBLUP on Exp II
├── build_web_model.py       compresses for deployment
├── requirements.txt
├── render.yaml
└── web_model.npz            produced by build_web_model.py
```

---

## Deploy on Render

1. Push this folder to a GitHub repository, **including `web_model.npz`**
   (about 0.2 MB, well under any file limit).
2. On [render.com](https://render.com): New → Web Service → connect the repo.
3. Render reads `render.yaml`; no manual configuration is needed.
4. First build takes 2–3 minutes. You get a URL like
   `https://deephybrid.onrender.com`.

The free tier sleeps after 15 minutes idle and takes ~30 seconds to wake.
Open the link a minute before a demo.

Railway and Fly.io work the same way with the same start command:

```
uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

## Why the model is compressed

Prediction only ever needs `alpha @ A_train`, never `A_train` itself:

```
pred = mu + (w_A/den_A)·(alpha @ A_train)·A_new
          + (w_D/den_D)·(alpha @ U_train)·U_new
```

Collapsing those products turns two 1725 × 8258 float32 matrices — about
114 MB in memory — into two 8,258-element vectors, roughly 0.2 MB. That is the
difference between fitting a 512 MB free tier and not.

This is an associativity rearrangement, not an approximation.
`build_web_model.py` asserts it: measured agreement is 1.8 × 10⁻¹⁵, which is
floating-point noise.

---

## API

| endpoint | purpose |
|---|---|
| `GET /api/parents` | selectable parents and headline counts |
| `POST /api/predict` | one cross → yield, percentile, confidence |
| `GET /api/rank` | ranked shortlist, filterable |
| `GET /api/distribution` | histogram across all possible crosses |
| `GET /api/methodology` | measured performance and reproduction checks |
| `GET /docs` | interactive documentation |

```bash
curl -X POST localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"female":"elixer","male":"zf040"}'
```

`/docs` is worth opening in a defence — it shows a real, documented interface
behind the visual design.

---

## What is not in the interface, and why

Earlier mockups included panels with no model behind them. Each is absent
rather than filled with a plausible-looking number.

| mockup element | why it is not here |
|---|---|
| Compatibility Index (93% OPTIMAL) | The model predicts yield, not whether a cross can be made. Feasibility depends on anther extrusion, flowering synchrony and height difference — fixed at trial design. The training data contains only successful crosses, so there are no failures to learn from. |
| Plant Health Scanner, pathogen detection | No disease model exists in this project. |
| Trait scores — drought, pest resistance, cycle time, fibre quality | None of these traits are in the dataset. Grain yield is the only phenotype. |
| 14.8 t/ha predicted yield | Observed Exp II hybrids average **10.08** Mg/ha (SD 0.35). 14.8 is outside the real range. |
| Temperature / rainfall sliders | Climate covariates take 5 unique values across the data — 2.3 bits, equivalent to a 5-level trial-identity factor. Using them would raise apparent accuracy by encoding which trial a plot came from. |
| 98% ACCURACY, real-time telemetry, IoT | Measured predictive ability is r = 0.73 at best. The rest has no data source. |

This matters beyond tidiness. The thesis shows the earlier r = 0.89 was an
artefact of evaluation design. An interface asserting a 93% compatibility index
and 98% accuracy would contradict that argument in front of an examiner who has
just read the limitations chapter.

---

## Design decisions

**The distribution ridge is the signature.** A single yield figure means little
when observed hybrids span 10.08 ± 0.35 Mg/ha and BLUP shrinks predictions
toward that mean. Showing where a cross falls among all 6,942 alternatives — one
gold bar in a field of grey — is the honest core of the product, so it is the
one element given visual weight.

**Ranking over raw value.** Pearson r measures how well a model *orders*
genotypes, so the interface reports percentile position. The reference spans
every combination, not only crosses that were grown — using only grown crosses
would measure against a set a breeder had already selected, biasing the
percentile upward.

**Observed crosses are labelled.** 1,720 of 6,942 combinations were actually
grown. For those the measured value is shown, and the user is told to trust it
over the estimate.

**Three decimal places.** Two would hide differences that matter at SD 0.35.

**Type and palette.** Fraunces for display, Inter for body, system mono for data
labels and axes. Deep green `#15402A`, wheat gold `#C08A1E`, cream `#FBFAF6` —
the palette from the approved mockups.

---

## Adding a logo

Save the logo as `static/logo.png`. The header picks it up automatically and
removes the image element if the file is missing, so nothing breaks either way.

---

## Tested

Served under Uvicorn and driven with a headless browser: all four pages render,
every endpoint returns 200, prediction and ranking round-trip correctly, and the
console is clean apart from the optional logo 404.

Source data: Zhao, Y., Thorwarth, P., Jiang, Y., et al. (2021). Unlocking big
data doubled the accuracy in predicting the grain yield in hybrid wheat.
*Science Advances* 7(24), eabf9106.
