# 🔄 Counterfactual Explanations — From Prediction to Action

Knowing a model predicts high risk is only half the story. The harder and more valuable question is: what would need to change to get a different outcome? Counterfactual explanations bridge the gap between a model's diagnosis and a patient's ability to act on it — translating a probability score into concrete, achievable intervention targets.

---

## 📌 Project Snapshot

| | |
|---|---|
| **Dataset** | Pima Indians Diabetes Dataset |
| **Model** | XGBoost Classifier |
| **Method** | Counterfactual explanation generation (implemented from scratch) |
| **Key Output** | Minimum feature changes to flip predicted risk from high to low |
| **Libraries** | `xgboost` · `scikit-learn` · `pandas` · `matplotlib` · `numpy` |

---

## 🧠 Counterfactuals vs SHAP

SHAP (Project 7) answers: *why did the model predict this?*
Counterfactuals answer: *what would need to change for the prediction to be different?*

Both are explainability tools — but they serve different purposes. SHAP is diagnostic, looking backward at what drove the current prediction. Counterfactuals are actionable, looking forward at what could change the outcome.

---

## 📐 What Makes a Good Counterfactual?

Not every set of changes qualifies. Four properties define a clinically useful counterfactual:

| Property | Requirement |
|----------|-------------|
| **Validity** | The proposed changes must actually flip the model's prediction |
| **Proximity** | Changes should be as small as possible |
| **Sparsity** | Fewer features changed is better — ideally 1 to 3 |
| **Actionability** | Only suggest changes to features the patient can actually control |

The actionability constraint is the most important and most often overlooked. Telling a patient to change their age or family history is valid and proximal — but completely useless. This project explicitly separates actionable features from fixed ones and only generates counterfactuals over the actionable set.

---

## 🎯 Actionable vs Fixed Features

| Feature | Actionable? | Reason |
|---------|-------------|--------|
| Glucose | ✅ Yes | Diet and medication |
| BMI | ✅ Yes | Diet and exercise |
| BloodPressure | ✅ Yes | Medication and lifestyle |
| Insulin | ✅ Yes | Medical intervention |
| SkinThickness | ✅ Yes | Proxy for body fat — modifiable |
| Pregnancies | ❌ No | Historical fact |
| Age | ❌ No | Cannot be changed |
| DiabetesPedigree | ❌ No | Genetic/family history — fixed |

---

## 🔧 Implementation

Counterfactual generation is implemented from scratch without external libraries — building intuition for how the search works before relying on packages.

**Single-feature search:** For each high-risk patient, systematically test values across the full allowed range of each actionable feature. Any value that flips the prediction below the target threshold (45%) is recorded as a valid single-feature counterfactual.

**Two-feature search:** If single-feature counterfactuals don't suffice, pairs of actionable features are searched simultaneously.

**Diversity:** Counterfactuals are selected to be as different from each other as possible — different feature combinations — so patients and clinicians have multiple intervention pathways to choose from rather than one.

**Ranking:** Valid counterfactuals are ranked by normalised total change (smaller = more achievable).

---

## 📊 Analysis Performed

For the 5 highest-risk patients in the test set, up to 3 diverse counterfactuals are generated per patient. For each counterfactual, the output records which features changed, by how much, in which direction, and what the new predicted risk is.

A final analysis asks: **which features appear most often across all generated counterfactuals?** These are the highest-leverage intervention targets — the features where change produces the most consistent risk reduction across patients.

---

## 📈 Visualisations Generated

| File | Description |
|------|-------------|
| `plot1_counterfactuals_comparison.png` | Before/after feature values for top patient — all 3 counterfactuals |
| `plot2_risk_reduction.png` | Risk reduction achieved per patient per counterfactual |
| `plot3_intervention_features.png` | Feature frequency across all generated counterfactuals |
| `plot4_patient_profile.png` | High-risk patient vs healthy and diabetic population averages |

---

## 🔍 Key Findings

Single-feature counterfactuals — changing just one measurement — are sufficient to flip predictions for most high-risk patients. **Glucose** is the most frequent intervention target, appearing in the majority of generated counterfactuals. This is consistent with the SHAP findings in Project 7, where glucose was also the dominant feature, and with clinical reality — glucose is the most direct and responsive marker in diabetes management.

The before/after visualisations make the clinical story concrete: a patient predicted at 82% risk needs their glucose to decrease from 168 to 127 — a clinically achievable target through dietary change — to drop below the 45% threshold. This is a meaningful, actionable recommendation rather than an abstract probability score.

---

## 📂 Repository Structure

```
counterfactual-explanations/
├── diabetes.csv
├── counterfactual_explanations.py
├── plot1_counterfactuals_comparison.png
├── plot2_risk_reduction.png
├── plot3_intervention_features.png
├── plot4_patient_profile.png
└── README.md
```

---

## ⚙️ Setup

```bash
git clone https://github.com/Shaflovescoffee19/counterfactual-explanations.git
cd counterfactual-explanations
pip3 install xgboost scikit-learn pandas matplotlib seaborn numpy
python3 counterfactual_explanations.py
```

---

## 📚 Skills Developed

- The conceptual difference between diagnostic explainability (SHAP) and actionable explainability (counterfactuals)
- Four properties of a valid counterfactual — validity, proximity, sparsity, actionability — and why all four matter
- Implementing a feature space search algorithm from scratch
- Generating diverse counterfactuals — multiple intervention pathways rather than a single solution
- Connecting model output to real-world intervention design
- The limits of ML explainability — a counterfactual tells you what to change, not whether the change is sufficient or safe in a clinical context

---

## 🗺️ Learning Roadmap

**Project 8 of 10** — a structured series building from data exploration through to advanced ML techniques.

| # | Project | Focus |
|---|---------|-------|
| 1 | Heart Disease EDA | Exploratory analysis, visualisation |
| 2 | Diabetes Data Cleaning | Missing data, outliers, feature engineering |
| 3 | Cancer Risk Classification | Supervised learning, model comparison |
| 4 | Survival Analysis | Time-to-event modelling, Cox regression |
| 5 | Customer Segmentation | Clustering, unsupervised learning |
| 6 | Gene Expression Clustering | High-dimensional data, heatmaps |
| 7 | Explainable AI with SHAP | Model interpretability |
| 8 | **Counterfactual Explanations** ← | Actionable predictions |
| 9 | Multi-Modal Data Fusion | Stacking, ensemble methods |
| 10 | Transfer Learning | Neural networks, domain adaptation |
