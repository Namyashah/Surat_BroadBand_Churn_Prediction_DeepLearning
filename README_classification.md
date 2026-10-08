![Plots Screenshot](images/Screenshot%202026-10-08%20194518.png)
# 📊 Surat DTH / Broadband Customer Churn Prediction

> 🧠 **Deep Learning — Neural Networks | Classification Project**
>
> A Feedforward Artificial Neural Network (ANN) project for predicting whether a Surat DTH/Broadband customer is likely to churn.

---

## 🎯 Problem Statement

A DTH/Broadband provider wants to identify customers who are likely to **churn**.

The dataset contains **5,000 customer records** with approximately **17% churn cases**, making churn the minority class.

The objective is to build a neural-network classification model that can identify churn customers while handling class imbalance effectively.

### 🎯 Target
- `0` → Not Churn
- `1` → Churn

---

## 🗂️ Dataset Features

| Feature | Description |
|---|---|
| `tenure_months` | Customer tenure |
| `contract_type` | Type of customer contract |
| `monthly_bill` | Monthly bill amount |
| `avg_data_usage_gb` | Average data usage |
| `complaint_count_6m` | Complaints in the last 6 months |
| `late_payment_count` | Number of late payments |
| `competitor_offer_nearby` | Competitor offer availability |
| `festival_discount_availed` | Festival discount usage |
| `customer_service_calls` | Customer service calls |
| `locality` | Customer locality |
| `lucky_draw_entries` | 🚫 Intentionally irrelevant feature |
| `signup_weekday` | 🚫 Intentionally irrelevant feature |
| `churn` | 🎯 Target variable |

---

# 🔎 Project Workflow

```text
Raw Customer Data
        ↓
Exploratory Data Analysis
        ↓
Class Balance Analysis
        ↓
Stratified Train/Test Split
        ↓
Categorical Encoding + Numerical Scaling
        ↓
Baseline ANN
        ↓
Baseline Evaluation
        ↓
SMOTE for Class Imbalance
        ↓
Improved ANN
        ↓
Dropout + Batch Normalization
        ↓
L2 Regularization + Early Stopping
        ↓
Threshold Tuning using Precision-Recall Curve
        ↓
Final Evaluation using ROC-AUC
        ↓
Irrelevant Feature Investigation
```

---

## 📌 Q8 — Class Balance Analysis

The first step was to understand the distribution of the target variable.

### What was investigated?
- Number of churn vs non-churn customers
- Percentage of each class
- Class imbalance
- Why plain accuracy can be misleading

Because churn is the minority class, a model could achieve high accuracy simply by predicting most customers as **non-churn**.

### 💡 Key Insight

If the majority class represents around 83% of the data, a model that predicts `0` for every customer could achieve approximately 83% accuracy while completely failing to identify churn customers.

Therefore, **Precision, Recall and F1-score are important**, especially for the churn class.

---

# ⚙️ Q9 — Data Preprocessing

The dataset contains both numerical and categorical variables.

### 🔧 Preprocessing performed

**Numerical Features**
- Standardized using `StandardScaler`

**Categorical Features**
- Encoded using `OneHotEncoder(handle_unknown="ignore")`

### 🧪 Train/Test Split

A **stratified split** was used so that the churn/non-churn ratio remains approximately consistent between training and testing data.

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

---

# 🤖 Q10 — Baseline ANN

A basic Feedforward Neural Network was created as the baseline model.

### Architecture

```text
Input
  ↓
Dense(64, ReLU)
  ↓
Dense(32, ReLU)
  ↓
Dense(1, Sigmoid)
```

### Configuration

- Optimizer: **Adam**
- Loss: **Binary Cross-Entropy**
- Batch size: **32**
- Epochs: **50**
- No class balancing
- No Dropout
- No Batch Normalization
- No L2 regularization
- No threshold tuning

The baseline provides a reference point for evaluating the improved model.

---

# 📈 Q11 — Baseline Evaluation

The baseline model was evaluated using:

- ✅ Accuracy
- ✅ Precision
- ✅ Recall
- ✅ F1-score
- ✅ Confusion Matrix

Special attention was given to **Recall for the churn class (`1`)**, because missing an actual churn customer is important for the business problem.

### Why not rely only on Accuracy?

With an imbalanced dataset, accuracy can hide poor minority-class performance.

A model can have good accuracy while having very low churn recall.

---

# 🚀 Q12 — Improved ANN

The baseline model was improved using **SMOTE** and multiple neural-network regularization techniques.

## 🧬 SMOTE

**Synthetic Minority Over-sampling Technique (SMOTE)** was applied to the **training data only**.

```text
Original Training Data
        ↓
      SMOTE
        ↓
Balanced Training Data
```

SMOTE generates synthetic minority-class examples to reduce the class imbalance.

⚠️ The test set was kept untouched to ensure fair evaluation.

---

## 🧠 Improved Architecture

```text
Input
  ↓
Dense(128, ReLU) + L2
  ↓
Batch Normalization
  ↓
Dropout(0.3)
  ↓
Dense(64, ReLU) + L2
  ↓
Batch Normalization
  ↓
Dropout(0.3)
  ↓
Dense(32, ReLU) + L2
  ↓
Dense(1, Sigmoid)
```

### Techniques Used

| Technique | Purpose |
|---|---|
| 🧬 SMOTE | Handles class imbalance |
| 💧 Dropout | Reduces overfitting |
| 📏 Batch Normalization | Stabilizes training |
| 🛡️ L2 Regularization | Controls model complexity |
| ⏹️ Early Stopping | Stops training when validation loss stops improving |

### Early Stopping

The model monitored validation loss and restored the best weights:

```python
EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)
```

---

# 🎚️ Q13 — Decision Threshold Tuning

The default classification threshold is usually:

```text
0.5
```

However, `0.5` is not necessarily the best threshold for an imbalanced churn problem.

The model first produces probabilities:

```text
0.82 → likely churn
0.31 → likely non-churn
0.57 → likely churn
```

A **Precision-Recall curve** was used to evaluate different thresholds.

### 🔍 Threshold Selection

F1-score was calculated for the available thresholds.

The threshold producing the **highest F1-score** was selected as the final threshold.

```python
y_pred_tuned = (
    y_prob_imp >= best_threshold
).astype(int)
```

### 📝 Justification

> **The threshold was selected because it achieved the highest F1-score, providing the best balance between precision and recall for the minority churn class.**

---

# 🏆 Q14 — Final Model Evaluation

The final model was evaluated using:

### ROC-AUC

ROC-AUC measures how effectively the model distinguishes between churn and non-churn customers.

```python
roc_auc = roc_auc_score(
    y_test,
    y_prob_imp
)
```

### Classification Report

The final tuned predictions were used to generate:

- Precision
- Recall
- F1-score
- Support

```python
classification_report(
    y_test,
    y_pred_tuned
)
```

### 📊 Final Evaluation Metrics

> Replace the values below with the actual results from your notebook.

| Metric | Final Model |
|---|---:|
| Accuracy | `YOUR_VALUE` |
| Precision | `YOUR_VALUE` |
| Recall | `YOUR_VALUE` |
| F1-score | `YOUR_VALUE` |
| ROC-AUC | `YOUR_VALUE` |
| Selected Threshold | `YOUR_VALUE` |

---

# 🚫 Irrelevant Feature Investigation

Two columns were intentionally included without predictive meaning:

```text
lucky_draw_entries
signup_weekday
```

Permutation importance was used to investigate whether the trained model relied on these features.

### Method

```python
permutation_importance(
    improved_model,
    x_test_preprocessed,
    y_test,
    scoring="roc_auc",
    n_repeats=5,
    random_state=42
)
```

### 💡 Interpretation

The importance values for the two irrelevant features should ideally be **close to zero**.

This indicates that the model does not significantly depend on these intentionally irrelevant variables when making churn predictions.

> ⚠️ Small negative permutation-importance values can occur and do not necessarily indicate a problem; they mean that shuffling the feature did not reduce the model's score.

---

# 📊 Evaluation Visualizations

The project includes/uses:

- 📉 Training vs Validation Loss
- 📈 Precision-Recall Curve
- 🎯 Confusion Matrix
- 📊 Classification Metrics
- 📈 ROC-AUC evaluation

These visualizations help understand both model training behavior and classification performance.

---

# 🧰 Technologies Used

| Technology | Purpose |
|---|---|
| 🐍 Python | Programming |
| 🐼 Pandas | Data handling |
| 🔢 NumPy | Numerical operations |
| 📊 Matplotlib | Visualization |
| 🤖 TensorFlow / Keras | Neural Network |
| 🧪 Scikit-learn | Preprocessing & evaluation |
| 🧬 Imbalanced-learn | SMOTE |

---

# 🧠 Key Learning Outcomes

Through this classification task, I practiced:

- Understanding class imbalance
- Stratified train/test splitting
- Encoding categorical variables
- Feature scaling
- Building a baseline ANN classifier
- Using Binary Cross-Entropy
- Handling imbalance with SMOTE
- Applying Dropout
- Applying Batch Normalization
- Applying L2 regularization
- Using Early Stopping
- Evaluating Precision, Recall and F1-score
- Understanding why accuracy can be misleading
- Tuning classification thresholds
- Using Precision-Recall curves
- Evaluating ROC-AUC
- Investigating irrelevant/decoy features using permutation importance

---

# 🔄 Complete Solution Summary

```text
🎯 Problem
Customer churn prediction
        ↓
🔎 EDA
Understand class imbalance
        ↓
⚙️ Preprocessing
Encoding + Scaling + Stratified Split
        ↓
🤖 Baseline ANN
Binary Cross-Entropy
        ↓
📊 Evaluation
Accuracy + Precision + Recall + F1
        ↓
🧬 SMOTE
Balance training data
        ↓
🚀 Improved ANN
Dropout + BatchNorm + L2
        ↓
⏹️ Early Stopping
Reduce overfitting
        ↓
🎚️ Threshold Tuning
Precision-Recall + F1
        ↓
🏆 Final Evaluation
ROC-AUC + Classification Report
        ↓
🚫 Decoy Analysis
Permutation Importance
```

---

## ⭐ Conclusion

The project developed a neural-network-based churn prediction system from data preprocessing through final model evaluation.

The baseline ANN established a performance reference, while the improved model addressed class imbalance using **SMOTE** and improved generalization using **Dropout, Batch Normalization, L2 Regularization, and Early Stopping**.

Finally, **decision-threshold tuning** was performed using the Precision-Recall curve to obtain a better precision-recall balance for the minority churn class.

> 🚀 **The complete workflow demonstrates how a basic ANN can be progressively improved to handle an imbalanced classification problem.**

---

### 👩‍💻 Project Focus

**Deep Learning | Neural Networks | Binary Classification | Customer Churn Prediction**
