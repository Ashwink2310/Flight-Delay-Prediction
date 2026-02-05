# Flight Delay Prediction System

A comprehensive machine learning system for predicting flight arrival delays using gradient boosting models with advanced feature engineering and SHAP-based interpretability.

## Project Overview

This project analyzes over 1.2 million flight records to predict arrival delays (15+ minutes) using ensemble machine learning techniques. The system achieves 0.618 validation AUC through sophisticated feature engineering, hyperparameter optimization, and handles severe class imbalance (5:1 ratio).

## Key Highlights

- **Dataset**: 1.2M+ flight records from January 2019 and January 2020
- **Models**: XGBoost, LightGBM, CatBoost with Optuna hyperparameter tuning
- **Best Performance**: XGBoost with 0.618 validation AUC, 0.589 test AUC
- **Features**: 21 engineered features from 7 base features using target encoding and temporal patterns
- **Interpretability**: SHAP analysis for feature importance and prediction explanations

## Technologies Used

- **Languages**: Python 3.12
- **ML Frameworks**: XGBoost, LightGBM, CatBoost, scikit-learn
- **Optimization**: Optuna (50 trials per model)
- **Interpretability**: SHAP
- **Data Processing**: Pandas, NumPy, cuDF (GPU-accelerated)
- **Visualization**: Matplotlib, Seaborn
- **Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling)

## Dataset

The project uses U.S. Department of Transportation flight data:
- **Source**: Bureau of Transportation Statistics (BTS)
- **Period**: January 2019 and January 2020
- **Records**: 1,191,331 total flights
- **Operational Flights**: 1,165,231 (after removing cancelled/diverted)
- **Target Variable**: ARR_DEL15 (binary: 1 = delayed ≥15 min, 0 = on-time)

**Note**: Dataset files are not included in this repository due to size constraints. Download from [BTS website](https://www.transtats.bts.gov/).

### Key Features
- **Temporal**: Day of week, day of month, departure time block, departure hour
- **Operational**: Carrier, origin airport, destination airport, route
- **Geographic**: Flight distance (with log and squared transformations)
- **Engineered**: Carrier-route combinations, time-airport interactions, historical delay rates

## Methodology

### 1. Exploratory Data Analysis
- Class distribution analysis (83.91% on-time, 16.09% delayed)
- Temporal patterns (delays increase throughout the day)
- Carrier and airport performance analysis
- Distance vs delay correlation

### 2. Feature Engineering
- **Target Encoding**: Out-of-fold encoding for high-cardinality features (airports, routes)
- **Temporal Features**: Hour of day, time-of-day categories, weekend indicators
- **Interaction Features**: Carrier-route, carrier-airport, time-location combinations
- **Distance Transformations**: Log transform, squared distance, categorical bins

### 3. Data Preprocessing
- Temporal train/val/test split (60%/20%/20%)
- SMOTE oversampling to address 5:1 class imbalance
- MinMax scaling for numerical features
- One-hot encoding for categorical features
- Final feature space: 162 features

### 4. Model Training
- **Baseline Models**: Majority class (0.500 AUC), Carrier average (0.544 AUC)
- **XGBoost**: 0.618 validation AUC, 0.589 test AUC
- **LightGBM**: 0.617 validation AUC, 0.593 test AUC
- **CatBoost**: 0.613 validation AUC, 0.587 test AUC
- Optuna hyperparameter tuning (50 trials per model)
- 5-fold stratified cross-validation (mean AUC: 0.745 ± 0.003)

### 5. Threshold Optimization
- Optimal threshold: 0.388 (F1-score optimized)
- Precision: 20.2%, Recall: 63.0%, F1: 30.5%
- Tradeoff analysis between false alarms and missed delays

## Key Results

### Model Performance
| Model | Validation AUC | Test AUC | Best Iteration |
|-------|---------------|----------|----------------|
| **XGBoost** | **0.618** | 0.589 | 400 |
| LightGBM | 0.617 | **0.593** | 600 |
| CatBoost | 0.613 | 0.587 | 165 |
| Carrier Baseline | - | 0.544 | - |

### Feature Importance (Top 5)
1. **OP_CARRIER_DL** (2.01%) - Delta Airlines indicator
2. **CARRIER_ROUTE_delay_rate** (1.69%) - Historical carrier-route delay rate
3. **DAY_CARRIER_4_YV** (1.43%) - Thursday + Mesa Airlines
4. **DAY_CARRIER_3_WN** (1.36%) - Wednesday + Southwest
5. **OP_CARRIER_WN** (1.36%) - Southwest Airlines indicator

### Insights
- **Temporal Patterns**: Delays increase throughout the day (early morning: lowest, evening: highest)
- **Carrier Impact**: Carrier identity and historical patterns are strongest predictors
- **Airport Influence**: Small regional airports (ASE, SWF) show highest delay rates (>30%)
- **Distance**: Minimal correlation with delays; operational factors dominate
- **Performance Degradation**: Validation-test gap suggests temporal distribution shift

## SHAP Analysis

SHAP (SHapley Additive exPlanations) provides model interpretability:
- **Feature Importance**: Validates carrier and route features as top predictors
- **Individual Predictions**: Waterfall plots explain correct predictions and errors
- **Interaction Effects**: Reveals how features combine to influence predictions

## Challenges & Limitations

1. **Class Imbalance**: 5:1 ratio (on-time vs delayed) requires careful handling
2. **Temporal Drift**: Declining delay rates across splits (18% → 11%) indicate non-stationarity
3. **Low Precision**: Only 20% of predicted delays are correct at optimal threshold
4. **Distribution Shift**: CV AUC (0.745) significantly exceeds test AUC (0.589)
5. **Missing Features**: Weather data, real-time conditions, aircraft positions not available

## Future Improvements

- [ ] Incorporate real-time weather data and forecasts
- [ ] Add aircraft tail number history and maintenance schedules
- [ ] Engineer network-level cascading delay features
- [ ] Implement carrier-specific and time-stratified models
- [ ] Explore deep learning architectures (LSTM, Transformers)
- [ ] Frame as regression problem (predict delay magnitude)
- [ ] Develop ensemble models combining multiple approaches
- [ ] Implement continuous retraining pipeline for production

## Requirements
```
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0
optuna>=3.3.0
shap>=0.42.0
imbalanced-learn>=0.11.0
scipy>=1.11.0
joblib>=1.3.0
```
---
