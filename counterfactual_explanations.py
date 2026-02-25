# ============================================================
# PROJECT 8: Counterfactual Explanations
# ============================================================
# WHAT THIS SCRIPT DOES:
#   1. Trains XGBoost on diabetes data
#   2. Identifies high-risk patients (predicted diabetic)
#   3. Generates counterfactual explanations from scratch
#      - Finds smallest feature changes to flip prediction
#      - Only changes actionable features (not age, genetics)
#      - Generates diverse counterfactuals per patient
#   4. Visualises counterfactuals as before/after comparisons
#   5. Ranks features by how often they appear in counterfactuals
#   6. Connects findings to clinical intervention design
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from itertools import product

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150
np.random.seed(42)

# ===========================================================
# STEP 1: LOAD, CLEAN, AND TRAIN MODEL
# ===========================================================

df = pd.read_csv("diabetes.csv")

# Clean impossible zeros (same as Projects 2 and 7)
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df_clean = df.copy()
for col in zero_cols:
    df_clean[col] = df_clean[col].replace(0, np.nan)
    df_clean[col] = df_clean.groupby("Outcome")[col].transform(
        lambda x: x.fillna(x.median())
    )

feature_names = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigree", "Age"
]

X = df_clean.drop("Outcome", axis=1)
X.columns = feature_names
y = df_clean["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = XGBClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, eval_metric="logloss", verbosity=0
)
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)

print("=" * 60)
print("STEP 1: MODEL TRAINED")
print("=" * 60)
print(f"  AUC-ROC : {auc:.4f}")
print(f"  High-risk patients (predicted >60%): {(y_prob > 0.6).sum()}")
print()

# ===========================================================
# STEP 2: DEFINE ACTIONABLE FEATURES
# ===========================================================
# These are features the patient CAN change through
# lifestyle, diet, or medical intervention.
# Fixed features (Age, Pregnancies, DiabetesPedigree)
# are excluded from counterfactual generation.

ACTIONABLE = {
    "Glucose":        {"min": 70,  "max": 200, "step": 5},
    "BloodPressure":  {"min": 50,  "max": 130, "step": 5},
    "BMI":            {"min": 18,  "max": 50,  "step": 1},
    "Insulin":        {"min": 10,  "max": 400, "step": 10},
    "SkinThickness":  {"min": 5,   "max": 60,  "step": 5},
}

FIXED = ["Pregnancies", "DiabetesPedigree", "Age"]

print("=" * 60)
print("STEP 2: ACTIONABLE FEATURES")
print("=" * 60)
print("  Can change (lifestyle/medical intervention):")
for feat, bounds in ACTIONABLE.items():
    print(f"    {feat:<18s}: range [{bounds['min']}, {bounds['max']}]")
print()
print("  Fixed (cannot change):")
for feat in FIXED:
    print(f"    {feat}")
print()

# ===========================================================
# STEP 3: COUNTERFACTUAL GENERATION FUNCTION
# ===========================================================
# For a high-risk patient, we search for the closest point
# in feature space that gets a prediction below 0.5.
# We only modify actionable features.
# We use a greedy approach: try small changes one feature
# at a time, then pairs, keeping the minimum total change.

def generate_counterfactuals(patient, model, actionable_features,
                              feature_names, n_counterfactuals=3,
                              target_prob=0.45):
    """
    Generate diverse counterfactual explanations for one patient.

    Parameters:
        patient           : pd.Series — one patient's feature values
        model             : trained classifier
        actionable_features: dict of feature bounds and steps
        feature_names     : list of all feature names
        n_counterfactuals : how many diverse CFs to find
        target_prob       : target prediction probability (below = success)

    Returns:
        list of dicts — each CF with changed values, new prediction,
        total change, and features modified
    """
    counterfactuals = []
    original_prob = model.predict_proba(
        patient.values.reshape(1, -1)
    )[0, 1]

    # Generate candidate counterfactuals by trying combinations
    # of changes to actionable features
    candidates = []

    # Single feature changes
    for feat in actionable_features:
        feat_idx = feature_names.index(feat)
        bounds = actionable_features[feat]
        original_val = patient[feat]

        # Try values across the allowed range
        test_values = np.arange(bounds["min"], bounds["max"] + 1, bounds["step"])

        for val in test_values:
            candidate = patient.copy()
            candidate[feat] = val
            new_prob = model.predict_proba(
                candidate.values.reshape(1, -1)
            )[0, 1]

            if new_prob < target_prob:
                # Normalised change (as fraction of feature range)
                norm_change = abs(val - original_val) / (bounds["max"] - bounds["min"])
                candidates.append({
                    "features_changed": [feat],
                    "original_values": {feat: original_val},
                    "new_values": {feat: val},
                    "new_prob": new_prob,
                    "total_change": norm_change,
                    "candidate": candidate
                })

    # Two-feature changes (if not enough single-feature CFs)
    if len(candidates) < n_counterfactuals:
        feat_list = list(actionable_features.keys())
        for i in range(len(feat_list)):
            for j in range(i + 1, len(feat_list)):
                feat1, feat2 = feat_list[i], feat_list[j]
                b1, b2 = actionable_features[feat1], actionable_features[feat2]
                orig1, orig2 = patient[feat1], patient[feat2]

                # Sample a subset of combinations for speed
                vals1 = np.arange(b1["min"], b1["max"] + 1, b1["step"] * 3)
                vals2 = np.arange(b2["min"], b2["max"] + 1, b2["step"] * 3)

                for v1, v2 in product(vals1, vals2):
                    candidate = patient.copy()
                    candidate[feat1] = v1
                    candidate[feat2] = v2
                    new_prob = model.predict_proba(
                        candidate.values.reshape(1, -1)
                    )[0, 1]

                    if new_prob < target_prob:
                        norm1 = abs(v1 - orig1) / (b1["max"] - b1["min"])
                        norm2 = abs(v2 - orig2) / (b2["max"] - b2["min"])
                        candidates.append({
                            "features_changed": [feat1, feat2],
                            "original_values": {feat1: orig1, feat2: orig2},
                            "new_values": {feat1: v1, feat2: v2},
                            "new_prob": new_prob,
                            "total_change": norm1 + norm2,
                            "candidate": candidate
                        })

    if not candidates:
        return []

    # Sort by total change (smallest change first = most actionable)
    candidates.sort(key=lambda x: x["total_change"])

    # Select diverse counterfactuals — ensure they change different features
    selected = []
    used_feature_sets = []

    for c in candidates:
        feat_set = frozenset(c["features_changed"])
        # Only add if this feature combination hasn't been selected yet
        if feat_set not in used_feature_sets:
            selected.append(c)
            used_feature_sets.append(feat_set)
        if len(selected) >= n_counterfactuals:
            break

    # If still need more, just take smallest changes
    if len(selected) < n_counterfactuals:
        for c in candidates:
            if c not in selected:
                selected.append(c)
            if len(selected) >= n_counterfactuals:
                break

    return selected

# ===========================================================
# STEP 4: GENERATE COUNTERFACTUALS FOR HIGH-RISK PATIENTS
# ===========================================================

print("=" * 60)
print("STEP 4: GENERATING COUNTERFACTUAL EXPLANATIONS")
print("=" * 60)

# Select top 5 highest-risk patients
high_risk_indices = np.argsort(y_prob)[-5:][::-1]
all_counterfactuals = {}

for idx in high_risk_indices:
    patient = X_test.iloc[idx].copy()
    original_prob = y_prob[idx]

    cfs = generate_counterfactuals(
        patient, model, ACTIONABLE, feature_names,
        n_counterfactuals=3
    )

    all_counterfactuals[idx] = {
        "patient": patient,
        "original_prob": original_prob,
        "true_label": y_test.iloc[idx],
        "counterfactuals": cfs
    }

    print(f"\n  Patient #{idx} — Original Risk: {original_prob*100:.1f}%")
    if cfs:
        for i, cf in enumerate(cfs, 1):
            feats = cf["features_changed"]
            changes = []
            for f in feats:
                orig = cf["original_values"][f]
                new  = cf["new_values"][f]
                changes.append(f"{f}: {orig:.1f} → {new:.1f}")
            print(f"    CF {i}: {' | '.join(changes)}")
            print(f"           New risk: {cf['new_prob']*100:.1f}% "
                  f"(reduction: {(original_prob - cf['new_prob'])*100:.1f}%)")
    else:
        print("    No counterfactual found within constraints")

print()

# ===========================================================
# STEP 5: VISUALISE — BEFORE/AFTER COMPARISON
# ===========================================================
# Pick the highest-risk patient and show all 3 CFs visually

focus_idx = high_risk_indices[0]
focus_data = all_counterfactuals[focus_idx]
focus_cfs  = focus_data["counterfactuals"]

if focus_cfs:
    n_cfs = min(3, len(focus_cfs))
    fig, axes = plt.subplots(1, n_cfs, figsize=(6 * n_cfs, 7))
    if n_cfs == 1:
        axes = [axes]

    for i, cf in enumerate(focus_cfs[:n_cfs]):
        changed_feats = cf["features_changed"]
        all_changed   = list(ACTIONABLE.keys())

        orig_vals = [focus_data["patient"][f] for f in all_changed]
        cf_vals   = []
        for f in all_changed:
            if f in cf["new_values"]:
                cf_vals.append(cf["new_values"][f])
            else:
                cf_vals.append(focus_data["patient"][f])

        x = np.arange(len(all_changed))
        width = 0.35

        bars1 = axes[i].bar(x - width/2, orig_vals, width,
                             label=f"Original ({focus_data['original_prob']*100:.0f}% risk)",
                             color="#C44E52", edgecolor="white", alpha=0.85)
        bars2 = axes[i].bar(x + width/2, cf_vals, width,
                             label=f"Counterfactual ({cf['new_prob']*100:.0f}% risk)",
                             color="#55A868", edgecolor="white", alpha=0.85)

        # Highlight changed features
        for j, feat in enumerate(all_changed):
            if feat in changed_feats:
                axes[i].axvspan(j - 0.5, j + 0.5,
                                alpha=0.12, color="gold")
                axes[i].text(j, max(orig_vals[j], cf_vals[j]) * 1.05,
                             "Changed", ha="center", fontsize=8,
                             color="darkorange", fontweight="bold")

        axes[i].set_xticks(x)
        axes[i].set_xticklabels(all_changed, rotation=25, ha="right", fontsize=9)
        axes[i].set_ylabel("Feature Value", fontsize=10)
        axes[i].set_title(
            f"Counterfactual {i+1}\n"
            f"Change: {', '.join(changed_feats)}\n"
            f"Risk: {focus_data['original_prob']*100:.0f}% → {cf['new_prob']*100:.0f}%",
            fontweight="bold", fontsize=10
        )
        axes[i].legend(fontsize=9)
        axes[i].grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Counterfactual Explanations for Patient #{focus_idx}\n"
        f"(Gold highlight = changed feature | Green = target state)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig("plot1_counterfactuals_comparison.png", bbox_inches="tight")
    plt.close()
    print("Saved: plot1_counterfactuals_comparison.png")

# ===========================================================
# STEP 6: RISK REDUCTION CHART
# ===========================================================
# For each high-risk patient, show how much each CF reduces risk

fig, ax = plt.subplots(figsize=(12, 6))

patient_labels = []
reductions = []
cf_descriptions = []
bar_colors = []
color_cycle = ["#4C72B0", "#DD8452", "#55A868"]

for idx in high_risk_indices:
    data = all_counterfactuals[idx]
    orig_prob = data["original_prob"]

    for i, cf in enumerate(data["counterfactuals"][:2]):
        reduction = (orig_prob - cf["new_prob"]) * 100
        desc = " + ".join(cf["features_changed"])
        patient_labels.append(f"P#{idx} CF{i+1}")
        reductions.append(reduction)
        cf_descriptions.append(desc)
        bar_colors.append(color_cycle[i % len(color_cycle)])

bars = ax.bar(patient_labels, reductions,
              color=bar_colors, edgecolor="white", alpha=0.9)

for bar, desc, val in zip(bars, cf_descriptions, reductions):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            desc, ha="center", va="bottom",
            fontsize=7.5, rotation=20, color="dimgray")
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"-{val:.0f}%", ha="center", va="center",
            fontsize=9, fontweight="bold", color="white")

ax.set_ylabel("Risk Reduction (%)", fontsize=12)
ax.set_xlabel("Patient + Counterfactual", fontsize=12)
ax.set_title("Risk Reduction Achieved by Each Counterfactual Intervention",
             fontsize=13, fontweight="bold")
ax.tick_params(axis="x", rotation=30)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("plot2_risk_reduction.png", bbox_inches="tight")
plt.close()
print("Saved: plot2_risk_reduction.png")

# ===========================================================
# STEP 7: MOST COMMON INTERVENTION FEATURES
# ===========================================================
# Which features appear most often in counterfactuals?
# These are the highest-leverage intervention targets.

print("=" * 60)
print("STEP 7: MOST COMMON INTERVENTION FEATURES")
print("=" * 60)

feature_counts = {feat: 0 for feat in ACTIONABLE}
total_cfs = 0

for idx, data in all_counterfactuals.items():
    for cf in data["counterfactuals"]:
        for feat in cf["features_changed"]:
            if feat in feature_counts:
                feature_counts[feat] += 1
        total_cfs += 1

feature_freq = pd.Series(feature_counts).sort_values(ascending=False)
print("  Feature frequency in counterfactuals:")
for feat, count in feature_freq.items():
    pct = count / total_cfs * 100 if total_cfs > 0 else 0
    print(f"    {feat:<18s}: {count} times ({pct:.0f}%)")
print()

fig, ax = plt.subplots(figsize=(9, 5))
colors_freq = ["#DD8452" if i < 2 else "#4C72B0"
               for i in range(len(feature_freq))]
feature_freq.plot(kind="bar", ax=ax,
                  color=colors_freq, edgecolor="white", alpha=0.9)
ax.set_xlabel("Feature", fontsize=12)
ax.set_ylabel("Times Appeared in Counterfactuals", fontsize=12)
ax.set_title("Most Common Intervention Targets Across All Patients\n"
             "(Higher = more often the key lever for risk reduction)",
             fontsize=12, fontweight="bold")
ax.tick_params(axis="x", rotation=25)
ax.grid(axis="y", alpha=0.3)
for i, v in enumerate(feature_freq.values):
    ax.text(i, v + 0.05, str(v), ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("plot3_intervention_features.png", bbox_inches="tight")
plt.close()
print("Saved: plot3_intervention_features.png")

# ===========================================================
# STEP 8: DETAILED PATIENT REPORT
# ===========================================================
# A clinical-style report for the top high-risk patient

focus_patient = all_counterfactuals[focus_idx]

print("=" * 60)
print("STEP 8: CLINICAL REPORT — HIGHEST RISK PATIENT")
print("=" * 60)
print(f"  Patient ID        : #{focus_idx}")
print(f"  True Diagnosis    : {'Diabetic' if focus_patient['true_label'] == 1 else 'Non-Diabetic'}")
print(f"  Predicted Risk    : {focus_patient['original_prob']*100:.1f}%")
print()
print("  Current Feature Values:")
for feat in feature_names:
    val = focus_patient["patient"][feat]
    actionable = "✓ actionable" if feat in ACTIONABLE else "✗ fixed"
    print(f"    {feat:<20s}: {val:6.1f}  ({actionable})")
print()

if focus_patient["counterfactuals"]:
    print("  Recommended Interventions:")
    for i, cf in enumerate(focus_patient["counterfactuals"], 1):
        reduction = (focus_patient["original_prob"] - cf["new_prob"]) * 100
        print(f"\n  Option {i} — Risk reduced by {reduction:.1f}%:")
        for feat in cf["features_changed"]:
            orig = cf["original_values"][feat]
            new  = cf["new_values"][feat]
            direction = "decrease" if new < orig else "increase"
            diff = abs(new - orig)
            print(f"    → {direction} {feat} from {orig:.1f} to {new:.1f} "
                  f"(change of {diff:.1f})")
        print(f"    New predicted risk: {cf['new_prob']*100:.1f}%")

# ===========================================================
# STEP 9: VISUALISE PATIENT PROFILE vs HEALTHY AVERAGE
# ===========================================================

healthy_avg = X_train[y_train == 0].mean()
diabetic_avg = X_train[y_train == 1].mean()
patient_vals = focus_patient["patient"]

actionable_feats = list(ACTIONABLE.keys())
x = np.arange(len(actionable_feats))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(x - width, [healthy_avg[f] for f in actionable_feats],
       width, label="Healthy Average", color="#55A868",
       edgecolor="white", alpha=0.8)
ax.bar(x, [patient_vals[f] for f in actionable_feats],
       width, label=f"Patient #{focus_idx} (Risk: {focus_patient['original_prob']*100:.0f}%)",
       color="#C44E52", edgecolor="white", alpha=0.8)
ax.bar(x + width, [diabetic_avg[f] for f in actionable_feats],
       width, label="Diabetic Average", color="#DD8452",
       edgecolor="white", alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(actionable_feats, rotation=20, ha="right", fontsize=10)
ax.set_ylabel("Feature Value", fontsize=12)
ax.set_title(f"Patient #{focus_idx} Profile vs Healthy and Diabetic Averages",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("plot4_patient_profile.png", bbox_inches="tight")
plt.close()
print()
print("Saved: plot4_patient_profile.png")

# ===========================================================
# FINAL SUMMARY
# ===========================================================

total_found = sum(
    len(d["counterfactuals"]) for d in all_counterfactuals.values()
)
avg_reduction = np.mean([
    (d["original_prob"] - cf["new_prob"]) * 100
    for d in all_counterfactuals.values()
    for cf in d["counterfactuals"]
]) if total_found > 0 else 0

print()
print("=" * 60)
print("PROJECT 8 COMPLETE — FINAL SUMMARY")
print("=" * 60)
print(f"  Dataset              : Pima Indians Diabetes")
print(f"  Model AUC            : {auc:.4f}")
print(f"  High-risk patients   : {len(high_risk_indices)}")
print(f"  Counterfactuals found: {total_found}")
print(f"  Avg risk reduction   : {avg_reduction:.1f}%")
print()
print(f"  Top intervention feature: {feature_freq.index[0]}")
print(f"  Most frequent lever for reducing diabetic risk prediction")
print()
print("  4 plots saved.")
print("  Ready to push to GitHub!")
print("=" * 60)
