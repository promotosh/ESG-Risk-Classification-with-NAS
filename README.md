# Optimizing ESG Risk Prediction with Neural Architecture Search (NAS)


---

## The Problem

ESG (Environmental, Social, and Governance) risk ratings are increasingly important for investors and companies. But most existing models — decision trees, logistic regression, random forests — were not designed to automatically find the best architecture for complex, structured ESG data.

This project asks: **can Neural Architecture Search (NAS) outperform traditional methods in classifying corporate ESG risk, and can we explain *why* the model makes the predictions it does?**

---

## The Data

- **Source:** RepRisk dataset via Wharton Research Data Services (WRDS)
- **Size:** 228,289 observations of real corporate ESG incidents
- **Target variable:** RepRisk Rating (RRR) — a letter-grade risk score from AAA (low risk) to D (very high risk)
- **Features used:** Current RRI, Peak RRI, Trend RRI, Country-Sector Average, Environmental %, Social %, Governance %
- **Coverage:** 28 ESG issue categories across Environmental, Social, Governance, and Cross-cutting themes

---

## What I Did

I applied two Neural Architecture Search methods to find the optimal deep learning architecture for this tabular ESG dataset — an unusual application, since NAS is typically used for image or speech data.

**Method 1: Random Search NAS** (primary method)  
Randomly samples neural network configurations across hyperparameter space. Used as the main method following Li & Talwalkar (2019), who showed random search is a strong and reproducible baseline.

**Method 2: Efficient Neural Architecture Search (ENAS)** (benchmark)  
Uses a controller trained with gradient descent to search for optimal subgraph architectures, with parameter sharing across candidate models.

**Hyperparameters explored:**
- Epochs: 100
- Batch sizes: 32, 64, 128, 256
- Hidden units: 64, 128, 256
- Learning rates: 0.001, 0.01, 0.1
- Activation functions: ReLU, Tanh, Sigmoid

**Infrastructure:** Amazon SageMaker (ml.c3.xlarge), PyTorch, JupyterLab on AWS

**Explainability:** Applied LIME (Local Interpretable Model-agnostic Explanations) to understand which ESG features drove individual predictions.

---

## Results

| Method | Best Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| ENAS | **99.95%** | 99.53% | 99.50% | 99.51% |
| Random Search NAS | 99.21% | 98.04% | 95.12% | 96.33% |

**Best ENAS architecture:** 64 hidden units, learning rate 0.001, Sigmoid activation  
**Best Random Search architecture:** 256 hidden units, learning rate 0.01, Sigmoid activation

ENAS was more consistent across trials (98.61% → 99.95%), while Random Search showed more variability (90.63% → 99.21%) — demonstrating that controlled search outperforms pure random sampling, though both converge to strong results.

---

## What the LIME Analysis Revealed

The LIME interpretability analysis showed which features actually drive ESG risk predictions:

- **Peak RRI** was the most influential feature — both positively and negatively across all risk classes. A company's worst recent ESG exposure is a strong signal of ongoing risk.
- **Country-sector average** was the second most important predictor, contributing up to 38% of prediction weight for some instances. Where a company operates matters enormously for ESG risk.
- **Environmental percentage** appeared consistently as a negative contributor — suggesting that environmental exposure alone does not drive high risk classifications without social/governance factors.
- **Social percentage** showed mixed effects depending on context — positive in some high-risk cases, negative in others.

This explainability layer is what separates this work from a black-box model. Investors and risk analysts can see *why* a company received a specific rating.

---

## Key Finding

> Random Search NAS, when properly tuned, performs comparably to the more complex ENAS method on tabular ESG data — supporting Li & Talwalkar (2019)'s finding that random search with early stopping is a competitive and reproducible baseline. This also demonstrates that NAS methods, traditionally applied to unstructured data, can be effectively extended to structured tabular datasets.

---

## Limitations

- Single random seed used — results may vary with different initializations
- Weight sharing in ENAS introduces noise to architecture evaluation
- Time-series dynamics of ESG risk not captured (static snapshot analysis)
- High accuracy may partly reflect the structured nature of RepRisk ratings

---

## Tech Stack

- **Python**, **PyTorch** — model development and NAS implementation
- **Amazon AWS SageMaker** — cloud training and hyperparameter tuning
- **LIME** — local explainability for individual predictions
- **WRDS / RepRisk** — enterprise-grade ESG data source

---

## Files

- `ESG MAIN CODE.ipynb` — full NAS implementation, training loops, LIME analysis
- `ESG OUTPUT HTML.html` — rendered notebook output with all results and visualizations

---

## Citation

Barua, P. (2024). *Optimizing ESG Risk Prediction with Neural Architecture Search (NAS): Improving Accuracy, Interpretability, and Explainability.* Master's Thesis, Norwegian School of Economics (NHH).
