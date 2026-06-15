# Optimizing ESG Risk Prediction with Neural Architecture Search (NAS)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Platform](https://img.shields.io/badge/Platform-AWS%20SageMaker-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

> **Master's Thesis** · Master of Science in Economics and Business Administration — Business Analytics  
> Norwegian School of Economics (NHH) · Bergen, Spring 2024  
> **Author:** Promotosh Barua · **Supervisor:** Geir Drage Berentsen

---

## Overview

This thesis applies **Neural Architecture Search (NAS)** to classify corporate ESG (Environmental, Social, and Governance) risk ratings — an uncommon application, since NAS is typically used for unstructured data like images, audio, and text. The study shows that NAS methods work effectively on structured tabular data, achieving over 99% accuracy on a large ESG dataset, while also explaining individual predictions using LIME.

**The core research question:**  
*Can NAS automatically find a better neural network architecture for ESG risk classification than traditional models — and can we explain why it makes each prediction?*

---

## Table of Contents

1. [Background](#background)
2. [Data](#data)
3. [Methods](#methods)
4. [Results](#results)
5. [LIME Explainability](#lime-explainability)
6. [Key Findings](#key-findings)
7. [Quick Start](#quick-start)
8. [Repository Structure](#repository-structure)
9. [Limitations](#limitations)
10. [Tech Stack](#tech-stack)
11. [Citation](#citation)

---

## Background

### What is NAS?

Neural Architecture Search (NAS) automates the process of designing a neural network. Instead of manually choosing the number of layers, hidden units, activation functions, and learning rates, NAS searches over many configurations to find the one that performs best on your data.

NAS has three core components:

| Component | Role |
|---|---|
| **Search Space** | Defines which architectures are possible (layer types, sizes, activations) |
| **Search Strategy** | How to explore that space (random, evolutionary, reinforcement learning) |
| **Performance Estimation** | How to evaluate each candidate architecture efficiently |

### Why apply NAS to ESG data?

Previous ML methods used on ESG data — decision trees, logistic regression, gradient boosting (XGB, LGBM), random forests — have limitations in scalability, performance, and interpretability, especially with large datasets. NAS removes the manual architecture-selection bottleneck and finds a provably better configuration for the specific dataset.

---

## Data

| Property | Detail |
|---|---|
| **Source** | RepRisk dataset via Wharton Research Data Services (WRDS) |
| **Observations** | 228,289 real corporate ESG incidents |
| **Target variable** | RepRisk Rating (RRR) — letter-grade risk score: AAA (low) → D (very high) |
| **ESG categories** | 28 issue categories across Environmental, Social, Governance, and Cross-cutting themes |

### Input Features

| Feature | Description |
|---|---|
| `current_RRI` | Company's current RepRisk Index — present-day ESG exposure level |
| `peak_RRI` | Highest RRI the company has ever reached — worst historical ESG exposure |
| `RRI_trend` | Direction of change over time — improving or worsening |
| `country_sector_average` | Baseline ESG risk for similar companies in the same region and industry |
| `environmental_percentage` | Share of incidents classified as Environmental |
| `social_percentage` | Share of incidents classified as Social |
| `governance_percentage` | Share of incidents classified as Governance |

---

## Methods

Two NAS methods were applied and compared:

### Method 1 — Random Search NAS (primary)

Randomly samples neural network configurations from a defined hyperparameter space and evaluates each one independently. Based on Li & Talwalkar (2019), who showed that random search with early stopping is a strong, reproducible baseline that is competitive with more advanced methods.

- Early stopping was used as a regularisation technique to prevent overfitting
- Multiple trials run across different hyperparameter combinations
- Results show sensitivity to hyperparameter choices — architecture selection matters

### Method 2 — ENAS: Efficient Neural Architecture Search (benchmark)

Uses a controller — itself a neural network — trained with gradient descent to search for optimal subgraph architectures. All candidate architectures share parameters, which reduces the computational cost of evaluating each one individually.

- More systematic and consistent than random search
- Controller learns which configurations to explore next
- Used as a benchmark to evaluate how close random search can get

### Hyperparameter Search Space

| Hyperparameter | Values Explored |
|---|---|
| Epochs | 100 |
| Batch size | 32, 64, 128, 256 |
| Hidden units | 64, 128, 256 |
| Learning rate | 0.001, 0.01, 0.1 |
| Activation function | ReLU, Tanh, Sigmoid |

**Infrastructure:** AWS SageMaker (ml.c3.xlarge), PyTorch, JupyterLab

---

## Results

### Summary

| Method | Best Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **ENAS** | **99.95%** | 99.53% | 99.50% | 99.51% |
| Random Search NAS | 99.21% | 98.04% | 95.12% | 96.33% |

### Trial-by-Trial Accuracy

| Trial | ENAS | Random Search |
|---|---|---|
| 1 | 98.61% | 90.63% |
| 2 | 99.84% | 98.51% |
| 3 | 96.10% | 93.21% |
| 4 | **99.95%** | **99.21%** |

ENAS was more consistent across trials (range: 96.10% → 99.95%).  
Random Search showed more variability (range: 90.63% → 99.21%).

### Best Architectures Found

| Method | Hidden Units | Learning Rate | Activation |
|---|---|---|---|
| ENAS | 64 | 0.001 | Sigmoid |
| Random Search | 256 | 0.01 | Sigmoid |

---

## LIME Explainability

LIME (Local Interpretable Model-agnostic Explanations) works by slightly varying input features and observing how the model's prediction changes — revealing which features drove each individual decision.

### LIME Results Table

| Prediction Probability | Risk Class | Top Feature | Contribution |
|---|---|---|---|
| 82% | 5 | `peak_RRI <= -1.09` | 13% |
| 82% | 5 | `country_sector_average = 0.82` | 8% |
| 82% | 5 | `environmental_percentage = -0.36` | 4% |
| 90% | 5 | `peak_RRI > 0.81` | 20% |
| 90% | 5 | `governance_percentage = -0.53` | 2% |
| 100% | 1 | `country_sector_average = -1.14` | 37% |
| 100% | 1 | `social_percentage > 1.15` | 4% |
| 96% | 1 | `country_sector_average` (−0.85 to high) | 29% |
| 96% | 1 | `peak_RRI <= -1.09` | 26% |
| 89% | 3 | `country_sector_average = 1.29` | 36% |
| 86% | 3 | `country_sector_average = 1.77` | 38% |

### What the LIME Analysis Reveals

| Feature | Pattern |
|---|---|
| **`peak_RRI`** | Most influential feature overall — a company's worst-ever ESG exposure is the strongest single predictor of current risk, appearing with both positive and negative contributions across risk classes |
| **`country_sector_average`** | Second most important — carries up to 38% of prediction weight; where a company operates and what sector it is in has a very large effect on risk classification |
| **`environmental_percentage`** | Consistently a **negative** contributor — environmental incidents alone do not push a company into high-risk without co-occurring social or governance factors |
| **`social_percentage`** | Mixed — positive in some high-risk predictions, negative in others, suggesting context-dependent interaction with other features |

This explainability layer is what separates the model from a black box. An investor or risk analyst can inspect any prediction and understand exactly which company-level signals triggered the classification.

---

## Key Findings

> Random Search NAS, when properly tuned with early stopping, performs comparably to the more complex ENAS method on structured tabular ESG data. This supports Li & Talwalkar (2019), who found that the performance gap between random search with early stopping and leading HPO methods is small. It also demonstrates that NAS — conventionally applied to unstructured data — can be effectively extended to structured financial data.

**Practical implication:** Start with Random Search NAS. It is simpler to implement and nearly as accurate. Use ENAS when you need greater consistency across multiple runs or have additional compute budget.

---

## Quick Start

### 1. Prerequisites

```bash
pip install torch torchvision numpy pandas scikit-learn lime jupyter
```

You will also need:
- Access to the **RepRisk dataset via WRDS** (paid subscription required)
- An **AWS SageMaker** environment (ml.c3.xlarge or equivalent) to reproduce training at scale

### 2. Clone the repository

```bash
git clone https://github.com/promotosh/esg-risk-classification-with-nas.git
cd esg-risk-classification-with-nas
```

### 3. Launch the notebook

```bash
jupyter notebook "ESG MAIN CODE.ipynb"
```

### 4. Notebook sections — run in order

| Section | What it does |
|---|---|
| Data loading & preprocessing | Loads RepRisk data, encodes features, prepares train/test split |
| Random Search NAS | Runs multiple trials across hyperparameter combinations, logs accuracy per config |
| ENAS | Trains a controller to search subgraph architectures with weight sharing |
| LIME analysis | Generates per-prediction feature importance scores for individual instances |

> **No data or cloud access?** Open `ESG OUTPUT HTML.html` in any browser to view all results, metrics, and visualisations from the pre-run notebook — no setup needed.

---

## Repository Structure

```
esg-risk-classification-with-nas/
├── ESG MAIN CODE.ipynb     # Complete pipeline: data prep, NAS training, LIME
└── ESG OUTPUT HTML.html    # Pre-rendered output with all results and plots
```

---

## Limitations

| Limitation | Detail |
|---|---|
| **Single random seed** | Results are specific to one initialisation; repeating with multiple seeds would improve robustness and generalisability |
| **ENAS weight sharing** | Shared parameters introduce noise into architecture evaluation and can cause ranking inconsistencies during search |
| **Static snapshot** | The model does not model ESG risk as a time series — it classifies based on current feature values without capturing how risk evolves over time |
| **Data access** | Reproducing results requires a paid WRDS / RepRisk subscription |
| **Accuracy ceiling** | Very high accuracy may partly reflect structural patterns in how RepRisk constructs its ratings, rather than pure predictive generalisability |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.8+ | Core language |
| PyTorch | Neural network training and NAS implementation |
| AWS SageMaker (ml.c3.xlarge) | Cloud training environment |
| LIME | Local prediction explainability |
| WRDS / RepRisk | ESG incident data source |
| JupyterLab | Notebook development environment |

---

## Abbreviations

| Abbreviation | Meaning |
|---|---|
| AI | Artificial Intelligence |
| AutoML | Automated Machine Learning |
| AWS | Amazon Web Services |
| DNN | Deep Neural Network |
| ENAS | Efficient Neural Architecture Search |
| ESG | Environmental, Social and Governance |
| HPO | Hyperparameter Optimisation |
| LGBM | Light Gradient Boosting Machine |
| LIME | Local Interpretable Model-agnostic Explanations |
| ML | Machine Learning |
| NAS | Neural Architecture Search |
| RRI | RepRisk Index |
| XGB | Extreme Gradient Boosting |

---

## Citation

Barua, P. (2024). *Optimizing ESG Risk Prediction with Neural Architecture Search (NAS): Improving Accuracy, Interpretability, and Explainability.* Master's Thesis, Master of Science in Economics and Business Administration — Business Analytics, Norwegian School of Economics (NHH), Bergen.
