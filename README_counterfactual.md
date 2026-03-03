# 🔄 Counterfactual Explanations — Actionable Diabetes Risk Reduction

A Machine Learning project that generates counterfactual explanations to find the smallest changes to a patient's actionable features that would flip their predicted diabetes risk from high to low. This is **Project 8 of 10** in my ML learning roadmap toward computational biology research.

---

## 📌 Project Overview

| Feature | Details |
|---|---|
| Dataset | Pima Indians Diabetes Dataset |
| Model | XGBoost Classifier |
| Technique | Counterfactual Explanation Generation (from scratch) |
| Key Output | Minimum feature changes to reduce predicted diabetes risk |
| Libraries | `xgboost`, `scikit-learn`, `pandas`, `matplotlib`, `numpy` |

---

## 🧠 What Are Counterfactuals?

A counterfactual explanation answers: **"What would need to change for the model to predict differently?"**

Unlike SHAP (Project 7) which explains *why* the current prediction was made, counterfactuals are **actionable** — they suggest specific interventions a patient and doctor can act on.

### Four Properties of a Good Counterfactual

| Property | Meaning |
|---|---|
| Validity | The changes must actually flip the prediction |
| Proximity | Changes should be as small as possible |
| Sparsity | Change as few features as possible (1-3 ideally) |
| Actionability | Only suggest changes to features the patient can control |

---

## 🎯 Actionable vs Fixed Features

| Feature | Actionable? | Reason |
|---|---|---|
| Glucose | ✅ Yes | Controlled through diet and medication |
| BMI | ✅ Yes | Controlled through diet and exercise |
| Blood Pressure | ✅ Yes | Controlled through medication and lifestyle |
| Insulin | ✅ Yes | Medical intervention available |
| Skin Thickness | ✅ Yes | Proxy for body fat — modifiable |
| Pregnancies | ❌ No | Historical fact |
| Age | ❌ No | Cannot be changed |
| Diabetes Pedigree | ❌ No | Genetic/family history — fixed |

---

## 📊 Visualisations Generated

| Plot | What It Shows |
|---|---|
| Before/After Comparison | Original vs counterfactual feature values for top patient |
| Risk Reduction Chart | How much each counterfactual reduces risk per patient |
| Intervention Features | Which features appear most often as intervention targets |
| Patient Profile | High-risk patient vs healthy and diabetic averages |

---

## 📂 Project Structure

```
counterfactual-explanations/
├── diabetes.csv                         # Dataset (from Project 2)
├── counterfactual_explanations.py       # Main script
├── plot1_counterfactuals_comparison.png
├── plot2_risk_reduction.png
├── plot3_intervention_features.png
├── plot4_patient_profile.png
└── README.md
```

---

## ⚙️ Setup Instructions

**1. Clone the repository**
```bash
git clone https://github.com/Shaflovescoffee19/counterfactual-explanations.git
cd counterfactual-explanations
```

**2. Install dependencies**
```bash
pip3 install xgboost scikit-learn pandas matplotlib seaborn numpy
```

**3. Add the dataset**
Copy `diabetes.csv` from Project 2 into this folder.

**4. Run the script**
```bash
python3 counterfactual_explanations.py
```

---

## 🔬 Connection to Research Proposal

This project directly implements the counterfactual component of **Aim 3** of a computational biology research proposal on CRC risk prediction in the Emirati population:

> *"Counterfactual explanations will reveal actionable interventions: raising a protective bacterial species from 15% to 25% abundance could decrease predicted risk from 18% to 13%"*

In the CRC context:
- **Actionable features** = modifiable microbiome taxa abundances (changed through diet, prebiotics, probiotics)
- **Fixed features** = genomic variants, age, sex
- The counterfactual pipeline is identical — only the features change

---

## 📚 What I Learned

- What **counterfactual explanations** are and how they differ from SHAP
- The four properties of a clinically useful counterfactual: validity, proximity, sparsity, actionability
- Why **actionable vs fixed features** must be separated in medical AI
- How to implement counterfactual search **from scratch** using greedy feature space exploration
- How **diverse counterfactuals** give patients multiple intervention pathways
- Why counterfactuals are essential for **regulatory approval** of clinical AI systems
- How counterfactual interventions translate to **real clinical recommendations**

---

## 🗺️ Part of My ML Learning Roadmap

| # | Project | Status |
|---|---|---|
| 1 | Heart Disease EDA | ✅ Complete |
| 2 | Diabetes Data Cleaning | ✅ Complete |
| 3 | Cancer Risk Classification | ✅ Complete |
| 4 | Survival Analysis | ✅ Complete |
| 5 | Customer Segmentation | ✅ Complete |
| 6 | Gene Expression Clustering | ✅ Complete |
| 7 | Explainable AI with SHAP | ✅ Complete |
| 8 | Counterfactual Explanations | ✅ Complete |
| 9 | Multi-Modal Data Fusion | 🔜 Next |
| 10 | Transfer Learning | ⏳ Upcoming |

---

## 🙋 Author

**Shaflovescoffee19** — building ML skills from scratch toward computational biology research.
