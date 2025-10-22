# ✈️ Flight Delay Prediction using Machine Learning (CatBoost + GPU)

## Overview
This project predicts whether a commercial U.S. flight will arrive late using data from the **Bureau of Transportation Statistics (BTS) On-Time Performance Dataset** for **January 2019 and 2020**.  
The goal is to forecast potential arrival delays **before departure**, leveraging only pre-departure features such as origin, destination, carrier, and time block.

---

## Key Features
- **Data Size:** ~933,000 flights (Jan 2019–2020 combined)  
- **Target Variable:** `ARR_DEL15` (1 = Delayed 15+ minutes, 0 = On-time)  
- **Class Imbalance:** ~80% on-time vs. ~20% delayed  

**Features Used**
- Temporal: `DAY_OF_WEEK`, `DAY_OF_MONTH`, `DEP_TIME_BLK`
- Geographic: `ORIGIN`, `DEST`
- Operational: `OP_CARRIER`, `DISTANCE`
- Engineered: Route-level smoothed delay rates (`ORIGIN_rate`, `DEST_rate`)

---

## Modeling Pipeline
1. **Data Preprocessing**
   - Removed cancelled and diverted flights  
   - Applied smoothed target encoding without leakage  
   - Created derived congestion-based features  
   - Scaled numeric values and one-hot encoded categoricals  

2. **Models Tested**
   - Logistic Regression (cuML GPU)
   - Random Forest (cuML GPU)
   - XGBoost (GPU)
   - CatBoost (GPU – Final Model)

3. **Evaluation Metrics**
   - ROC–AUC (primary metric)
   - Precision, Recall, F1-score
   - Confusion Matrix
   - Threshold tuning based on validation F1 optimization

---

## Results Summary

| Model | Validation AUC | Test AUC | Notes |
|:------|:--------------:|:--------:|:------|
| cuML Logistic Regression | 0.592 | 0.56 | Strong baseline; good recall balance |
| cuML Random Forest | 0.600 | 0.57 | Higher precision, weaker recall |
| XGBoost (GPU) | 0.611 | 0.57 | Balanced performance, slower |
| **CatBoost (GPU)** | **0.612** | **0.571** | **Best overall, interpretable, efficient** |

---


## Key Takeaways
- Even without weather or congestion data, **route and carrier-level features** provide meaningful predictive power.  
- **CatBoost’s categorical handling** outperformed other tree-based models in both stability and speed.  
- GPU acceleration via **RAPIDS cuML** and **CatBoost** reduced runtime by over **70%** compared to CPU models.  

---

## Future Work
- Integrate **real-time weather** and **airport congestion** data.  
- Test **temporal or sequence-based models** (e.g., RNNs or temporal boosting).  
- Develop a **REST API or dashboard** for flight delay alerts.

