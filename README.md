# ESG Risk Classification with Neural Architecture Search (NAS)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Platform](https://img.shields.io/badge/Platform-AWS%20SageMaker-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

> Master's Thesis — Norwegian School of Economics (NHH) · Partha Barua, 2024

This project classifies corporate ESG (Environmental, Social, Governance) risk ratings using Neural Architecture Search (NAS) — automatically finding the best neural network design for the task, rather than hand-tuning it. It also explains each prediction using LIME, so the model is not a black box.

---

## What This Project Does

1. **Loads** 228,289 real corporate ESG incident records from RepRisk (via WRDS)
2. **Searches** for the best neural network architecture using two NAS methods
3. **Trains** the winning architecture to classify ESG risk grades (AAA → D)
4. **Explains** each prediction with LIME — showing which features drove the decision

If you work in ESG analytics, sustainable finance, or ML research, this gives you a reproducible pipeline that achieves >99% accuracy on structured ESG data and tells you *why* it made each call.

---

## Quick Start

### 1. Install dependencies

```bash
pip install torch torchvision numpy pandas scikit-learn lime jupyter
```

### 2. Clone and open the notebook

```bash
git clone https://github.com/promotosh/esg-risk-classification-with-nas.git
cd esg-risk-classification-with-nas
jupyter notebook "ESG MAIN CODE.ipynb"
```

### 3. Run notebook sections in order

| Section | What it does |
|---|---|
| Data loading & preprocessing | Cleans and encodes RepRisk features |
| Random Search NAS | Randomly samples and trains network configs |
| ENAS | Trains a controller to intelligently search architectures |
| LIME analysis | Generates per-prediction feature importance explanations |

> **No data access?** Open `ESG OUTPUT HTML.html` in any browser to view all results and visualizations without re-running anything.

---

## Data

| Property | Value |
|---|---|
| Source | RepRisk via Wharton Research Data Services (WRDS) |
| Rows | 228,289 corporate ESG incident observations |
| Target | RepRisk Rating (RRR): AAA, AA, A, BBB, BB, B, CCC, CC, C, D |
| Features | Current RRI, Peak RRI, Trend RRI, Country-Sector Average, Environmental %, Social %, Governance % |
| ESG categories | 28 issue categories across Environmental, Social, Governance, and Cross-cutting themes |

**What the features mean:**
- **Current RRI** — the company's current RepRisk Index score (exposure level right now)
- **Peak RRI** — the highest RRI the company has ever reached (worst historical exposure)
- **Trend RRI** — direction of change over time (improving or worsening)
- **Country-Sector Average** — baseline risk for similar companies in the same region and industry
- **E / S / G %** — share of incidents falling into Environmental, Social, or Governance categories

---

## Methods

### Why NAS instead of picking a model manually?

Traditional model selection means trying a handful of architectures and picking the best one by hand. NAS automates this: it systematically searches the space of possible neural network configurations and finds the one that performs best on your data.

### Method 1 — Random Search NAS (primary)

Randomly samples neural network configurations from a defined search space and evaluates each one. Straightforward, fast, and surprisingly effective. Based on Li & Talwalkar (2019), who showed it is a strong reproducible baseline.

### Method 2 — ENAS (benchmark)

Efficient Neural Architecture Search trains a controller (also a neural network) that learns which architectures to try next. Shares weights across candidate models to avoid training each one from scratch. More sophisticated than random search, but also more complex to implement and tune.

### Search space

| Hyperparameter | Options |
|---|---|
| Hidden units | 64, 128, 256 |
| Learning rate | 0.001, 0.01, 0.1 |
| Activation function | ReLU, Tanh, Sigmoid |
| Batch size | 32, 64, 128, 256 |
| Epochs | 100 |

**Infrastructure:** AWS SageMaker (ml.c3.xlarge instance), PyTorch, JupyterLab

---

## Results

| Method | Accuracy | Precision | Recall | F1-Score | Consistency |
|---|---|---|---|---|---|
| ENAS | **99.95%** | 99.53% | 99.50% | 99.51% | High (98.61% → 99.95%) |
| Random Search NAS | 99.21% | 98.04% | 95.12% | 96.33% | Variable (90.63% → 99.21%) |

**Best ENAS config:** 64 hidden units · LR 0.001 · Sigmoid activation

**Best Random Search config:** 256 hidden units · LR 0.01 · Sigmoid activation

ENAS was more consistent across trials, but both methods converge to strong results. The gap narrows when Random Search is run with enough trials.

---

## LIME Explainability

LIME (Local Interpretable Model-agnostic Explanations) works by perturbing the input features slightly and observing how the model's prediction changes. This tells you which features mattered most for *each individual prediction*.

**What the analysis found:**

| Feature | Role in predictions |
|---|---|
| **Peak RRI** | Most influential feature — a company's worst-ever ESG exposure is the strongest predictor of current risk |
| **Country-Sector Average** | Second most important — where a company operates carries up to 38% of prediction weight |
| **Environmental %** | Consistently a negative contributor — environmental incidents alone don't push a company into high-risk without social/governance co-occurrence |
| **Social %** | Mixed — positive in some high-risk cases, negative in others depending on context |

This means an investor or risk analyst can look at any flagged company and understand exactly which signals triggered the high-risk classification — not just that the model predicted it.

---

## Key Takeaway

> Random Search NAS, when given enough trials, matches the performance of ENAS on structured tabular ESG data. This supports Li & Talwalkar (2019) and shows that NAS — typically used for image and speech tasks — transfers well to structured financial data.

If you are applying this to a new dataset, start with Random Search NAS. It is simpler to run and nearly as accurate. Use ENAS if you need maximum consistency across runs or have compute budget to spare.

---

## Limitations

- **Single seed** — results will vary with different random seeds; run multiple seeds to confirm stability
- **ENAS weight sharing** — shared weights across candidates introduces noise in individual architecture evaluations
- **Static snapshot** — the model does not capture how a company's ESG risk evolves over time (no time-series modeling)
- **Data dependency** — RepRisk data requires a paid WRDS subscription; results cannot be reproduced without access

---

## Repository Structure

```
esg-risk-classification-with-nas/
├── ESG MAIN CODE.ipynb     # Full pipeline: preprocessing, NAS, training, LIME
└── ESG OUTPUT HTML.html    # Pre-rendered notebook with all results and plots
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.8+ | Core language |
| PyTorch | Neural network training and NAS implementation |
| AWS SageMaker | Cloud training (ml.c3.xlarge) |
| LIME | Local prediction explainability |
| WRDS / RepRisk | ESG incident data source |
| JupyterLab | Notebook environment |

---

## Citation

Barua, P. (2024). *Optimizing ESG Risk Prediction with Neural Architecture Search (NAS): Improving Accuracy, Interpretability, and Explainability.* Master's Thesis, Norwegian School of Economics (NHH).
