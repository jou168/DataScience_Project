# Rainfall Prediction Using Ensemble Methods

## Project Overview

This project builds a robust binary classifier to predict daily rainfall using environmental data. The implementation uses an ensemble approach combining Random Forest, AdaBoost, and Logistic Regression with soft voting to achieve stable predictions.

## Team Members

- Andy Su
- George Barrone
- Brandon Jou
- Hang Bao
- Luciano Aldana

## Important: Git and Jupyter Notebooks

**Before committing notebook changes:**

Jupyter notebooks store outputs (cell results, plots, images) which can cause merge conflicts and bloat the repository. Always clear outputs before staging your changes:

This prevents:

- Large diffs full of base64-encoded images
- Merge conflicts from re-running cells
- Unnecessary repository bloat

## Dataset

### Training Data (`train.csv`)

- **Size**: 2,190 samples
- **Features**: 10 environmental variables
- **Target**: Binary rainfall indicator (0 = No Rain, 1 = Rain)
- **Class Distribution**: 75.3% rain, 24.7% no rain (imbalanced)

### Test Data (`test.csv`)

- **Size**: 730 samples
- **Features**: Same 10 environmental variables (no labels)

### Features

| Feature         | Description                  |
| --------------- | ---------------------------- |
| `pressure`      | Atmospheric pressure         |
| `maxtemp`       | Maximum daily temperature    |
| `temparature`   | Current temperature          |
| `mintemp`       | Minimum daily temperature    |
| `dewpoint`      | Dewpoint temperature         |
| `humidity`      | Relative humidity percentage |
| `cloud`         | Cloud cover percentage       |
| `sunshine`      | Hours of sunshine            |
| `winddirection` | Wind direction in degrees    |
| `windspeed`     | Wind speed                   |

## Methodology

### 1. Data Preprocessing

- Drop non-predictive features (`id`, `day`)
- Feature engineering:
  - Temperature range (diurnal variation)
  - Dewpoint depression (saturation indicator)
  - Temperature deviation from daily average
  - Humidity-dewpoint interaction term
- Feature standardization using StandardScaler
- Stratified train/validation split (80/20)

### 2. Handling Class Imbalance

- Use `class_weight='balanced'` parameter in all classifiers
- Maintain stratification in train/validation split
- Evaluate using F1-score (primary) in addition to accuracy

### 3. Model Ensemble

Three base classifiers combined using soft voting:

1. **Random Forest**

   - Handles non-linear relationships and feature interactions
   - Robust to outliers
   - Uses balanced class weights

2. **AdaBoost**

   - Focuses on difficult-to-classify samples
   - Adaptive boosting improves iteratively
   - Complements Random Forest's bagging approach

3. **Logistic Regression**
   - Captures linear relationships
   - Provides probability calibration
   - Fast and interpretable

### 4. Evaluation Metrics

- **Primary**: F1-score (balances precision and recall for imbalanced data)
- **Secondary**: Precision, Recall, ROC-AUC, Accuracy
- **Validation**: 5-fold stratified cross-validation

## Project Structure

```
Project/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── train.csv                      # Training data
├── test.csv                       # Test data
├── rainfall_prediction.ipynb      # Main Jupyter notebook
└── predictions.csv                # Final predictions (generated)
```

## Implementation

### Notebook Structure

The `rainfall_prediction.ipynb` notebook is organized into sections:

1. **Part 1: Data Loading & EDA**

   - Load datasets and examine structure
   - Statistical summaries and visualizations
   - Feature distribution analysis
   - Correlation analysis
   - Class imbalance visualization

2. **Part 2: Data Preprocessing**

   - Feature engineering
   - Feature scaling
   - Train/validation split
   - Data quality verification

3. **Part 3: Individual Models** (To be implemented)

   - Random Forest training and tuning
   - AdaBoost training and tuning
   - Logistic Regression training and tuning

4. **Part 4: Ensemble Integration** (To be implemented)

   - Voting classifier configuration
   - Cross-validation
   - Performance comparison

5. **Part 5: Final Predictions** (To be implemented)
   - Generate test set predictions
   - Create submission file
   - Model interpretation

## Setup

### Installation

```bash
pip install -r requirements.txt
```

### Required Packages

- pandas >= 2.0.0
- numpy >= 1.20.0
- matplotlib >= 3.4.0
- seaborn >= 0.13.0
- scipy >= 1.9.0
- scikit-learn >= 1.3.0
- jupyter >= 1.0.0

### Running the Notebook

```bash
jupyter notebook rainfall_prediction.ipynb
```

## Team Workflow

This project is divided among 5 team members:

| Person   | Responsibility                           | Status         |
| -------- | ---------------------------------------- | -------------- |
| Andy Su  | Data Loading & EDA                       | ✅ Complete    |
| Andy Su  | Preprocessing & Feature Engineering      | ✅ Complete    |
| Person 3 | Random Forest & AdaBoost Models          | 🔄 In Progress |
| Person 4 | Logistic Regression & Imbalance Handling | 🔄 In Progress |
| Person 5 | Ensemble Integration & Final Predictions | ⏳ Pending     |

## Key Insights

- **Class Imbalance**: 3:1 ratio requires balanced weights
- **Feature Correlations**: Strong multicollinearity among temperature variables
- **Engineered Features**: Dewpoint depression shows strong correlation with rainfall
- **Ensemble Rationale**: Combining tree-based (RF, AdaBoost) and linear (LR) models captures both non-linear and linear patterns

## Results

_TBD_

## Future Improvements

- Hyperparameter optimization using RandomizedSearchCV
- Additional feature engineering (polynomial features, cyclical encoding for wind direction)
- Alternative ensemble methods (stacking, weighted voting)
- SMOTE for synthetic minority oversampling
- Model interpretability analysis (SHAP values, feature importance)

## References

- "Bagging voting" refers to soft voting ensemble method
- Evaluation focuses on F1-score due to class imbalance, not just accuracy
