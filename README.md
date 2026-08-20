# 📉 Customer Churn Prediction — Who's About to Leave, and Why It Matters

> Losing a customer costs 5x more than keeping one. This project builds a model that flags at-risk telecom customers *before* they walk out the door.

## 🎯 The Problem

A telecom company has ~7,000 customers. Roughly 1 in 4 of them will churn. The company doesn't know *which* 1 in 4 — until it's too late to do anything about it. That's an expensive blind spot, and it's exactly the kind of problem a classification model is built to solve.

## 🔍 What I Did

- **Cleaned and prepped real-world messy data** — handled a subtly broken numeric column (`TotalCharges` stored as text with blank entries), encoded 15 categorical features, and dropped identifiers with zero predictive value.
- **Diagnosed class imbalance early.** Only **26.5% of customers actually churn** — train a model naively on this and it'll just learn to always predict "no churn" and still look "accurate." Applied **SMOTE (Synthetic Minority Oversampling)** to balance the training set before modeling, not after.
- **Let the data pick the algorithm.** Benchmarked Decision Tree, Random Forest, and XGBoost using 5-fold cross-validation — no assumptions, just evidence. Random Forest came out on top at **90% cross-validation accuracy**.
- **Validated honestly on unseen data.** Held out a proper test set (not touched during training or tuning) and evaluated with a full classification report — 78% accuracy, with precision and recall reported separately for both classes, not just a single vanity metric.
- **Built it to actually be usable.** Packaged the trained model and all label encoders into a portable, pickled prediction system — feed it a new customer's profile, get back a churn prediction *and* a probability score, ready to plug into a retention workflow.

## 📊 Results at a Glance

| Model | Cross-Validation Accuracy |
|---|---|
| Decision Tree | 78% |
| **Random Forest** | **90%** |
| XGBoost | 89% |

**On held-out test data:** 78% accuracy, with the model reasonably balancing precision and recall across both the "churn" and "no churn" classes — a harder, more honest bar than training accuracy alone.

## 🛠️ Tech Stack

`Python` · `Pandas` · `Scikit-learn` · `XGBoost` · `imbalanced-learn (SMOTE)` · `Pickle`

## 💡 Why This Matters

A churn model is only useful if a business can act on it. This project doesn't stop at "here's an accuracy score" — it ends with a deployable prediction system that takes a customer's profile and returns an actionable probability, the kind of output a retention team could realistically plug into a dashboard or CRM trigger.

## 🚀 Future Work

- Hyperparameter tuning (GridSearchCV) on the winning model for a further accuracy boost
- Try stratified k-fold cross-validation for more robust imbalance handling
- Explore downsampling as an alternative to SMOTE and compare trade-offs
- Add SHAP-based explainability so retention teams understand *why* a customer is flagged, not just that they are

---
*Built as an end-to-end ML classification project — from raw CSV to a deployable predictive system.*
