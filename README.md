# 🩺 Diabetes Prediction System

> Machine Learning solution for predicting diabetes risk using ensemble methods and feature engineering.

**Kaggle Competition**: [Playground Series - Season 5, Episode 12](https://www.kaggle.com/competitions/playground-series-s5e12)

**Score**: 0.70102 (ROC-AUC)

---

## 📊 About the Competition

This project is part of the **Kaggle Playground Series Season 5 Episode 12** competition focused on diabetes prediction. The competition challenges participants to build accurate models for predicting diabetes diagnosis using various health indicators and lifestyle factors.

### Competition Details

- **Competition**: Playground Series S5E12 - Diabetes Prediction
- **Task**: Binary classification (Diabetes: Yes/No)
- **Metric**: ROC-AUC Score
- **Dataset**: ~700,000 training samples, ~300,000 test samples
- **Features**: 17 health and lifestyle indicators

### My Approach & Results

**Achieved Score**: **0.70102** (ROC-AUC)

**Key Techniques Used**:

- ✅ **Ensemble Learning**: Combined 3 gradient boosting algorithms
  - LightGBM (weight: 0.25)
  - XGBoost (weight: 0.30)
  - CatBoost (weight: 0.45)
- ✅ **5-Fold Cross-Validation**: Stratified K-fold for robust evaluation
- ✅ **Feature Engineering**: Created 25+ derived features from domain knowledge
- ✅ **Weighted Ensemble**: Optimized model weights based on validation performance

---

## 🎯 Project Features


### Machine Learning Pipeline

- Data ingestion with quality checks
- Advanced feature engineering (cardiovascular, lipid profile, lifestyle metrics)
- Multi-model training with cross-validation
- Ensemble prediction with weighted averaging
- Comprehensive evaluation metrics


## 📁 Project Structure

```text
Kaggel/
├── src/                    # ML Pipeline
│   ├── main.py            # CLI entry point
│   ├── config.py          # Configuration
│   ├── ingest.py          # Data loading & validation
│   ├── features.py        # Feature engineering (25+ features)
│   ├── train.py           # Model training (5-fold CV)
│   ├── models.py          # Model initialization
│   ├── ensemble.py        # Ensemble predictions
│   └── evaluate.py        # Performance metrics
│
├── data/
│   ├── raw/             # Original Kaggle data
│   └── processed/       # Cleaned data
│
├── artifacts/
│   ├── models/          # Trained models (15 total)
│   └── submissions/     # Kaggle submissions
│
├── notebook/            # Exploratory Data Analysis
│
├── Dockerfile           # Container definition
├── docker-compose.yml   # Orchestration config
└── requirements.txt     # Python dependencies
```


## 🧠 Feature Engineering

### Input Features (17)

From the Kaggle competition dataset:

- **Demographics**: Age, Gender
- **Physical**: BMI, Waist-to-Hip Ratio
- **Cardiovascular**: Systolic BP, Diastolic BP, Heart Rate
- **Lipid Profile**: Total Cholesterol, LDL, HDL, Triglycerides
- **Lifestyle**: Physical Activity, Screen Time, Sleep Duration
- **Medical History**: Hypertension, Cardiovascular Disease, Family History

### Engineered Features (+25)

Domain-driven feature creation:

- **Cardiovascular Metrics**: Pulse pressure, Mean arterial pressure, Rate-pressure product
- **Lipid Ratios**: LDL/HDL ratio, Cholesterol/HDL ratio, TG/HDL ratio, Non-HDL cholesterol
- **Lifestyle Scores**: Activity-age interaction, Screen-activity ratio, Lifestyle risk score
- **Risk Combinations**: Age-BMI risk, Genetic history risk, Composite risk scores

**Total Features**: 42 (17 original + 25 engineered)

---

## 🎯 Model Architecture

### Ensemble Strategy

```text
┌─────────────────────────────────────┐
│     Input: 17 Features              │
└──────────────┬──────────────────────┘
               │
               ↓
┌─────────────────────────────────────┐
│  Feature Engineering → 42 Features   │
└──────────────┬──────────────────────┘
               │
               ↓
┌─────────────────────────────────────┐
│        5-Fold Cross-Validation       │
└──────────────┬──────────────────────┘
               │
     ┌─────────┼─────────┐
     ↓         ↓         ↓
┌─────────┐ ┌─────────┐ ┌─────────┐
│ LightGBM│ │ XGBoost │ │ CatBoost│
│ 5 folds │ │ 5 folds │ │ 5 folds │
│ ×0.25   │ │ ×0.30   │ │ ×0.45   │
└────┬────┘ └────┬────┘ └────┬────┘
     │           │           │
     └───────────┼───────────┘
                 ↓
     ┌───────────────────────┐
     │  Weighted Ensemble     │
     │  Final Prediction      │
     └───────────────────────┘
```

### Models Configuration

#### LightGBM

- Estimators: 800
- Learning Rate: 0.03
- Subsample: 0.9
- Ensemble Weight: 0.25

#### XGBoost

- Estimators: 800
- Learning Rate: 0.03
- Max Depth: 6
- Tree Method: hist
- Ensemble Weight: 0.30

#### CatBoost

- Iterations: 800
- Learning Rate: 0.03
- Depth: 6
- Loss Function: Logloss
- Ensemble Weight: 0.45

**Total Models**: 15 (3 algorithms × 5 folds)

**Threshold**: 0.55 for binary classification

---

## 📈 Competition Results

**Submission Score**: 0.70102 (ROC-AUC)

### Key Insights

1. **Feature Engineering Impact**: Engineered features (especially lipid ratios and cardiovascular metrics) significantly improved model performance
2. **Ensemble Benefits**: Weighted ensemble outperformed individual models by ~2-3%
3. **Cross-Validation**: 5-fold CV ensured robust generalization to test data
4. **Model Diversity**: Combining LightGBM, XGBoost, and CatBoost captured different patterns

### Performance Metrics

- **ROC-AUC**: 0.70102 (competition metric)
- **Precision-Recall AUC**: Used for threshold tuning
- **F1-Score**: Evaluated at optimal threshold (0.55)
- **Cross-Validation**: Consistent across all 5 folds


---

## 🛠️ Tech Stack

### Machine Learning

- Python 3.10+
- LightGBM, XGBoost, CatBoost
- Scikit-learn
- NumPy, Pandas

### DevOps

- Docker
- Docker Compose

## 📊 Dataset Information

**Source**: [Kaggle Playground Series S5E12](https://www.kaggle.com/competitions/playground-series-s5e12)

### Training Data

- Samples: ~700,000 rows
- Size: 79 MB
- Features: 17 + target
- Class Distribution: Imbalanced

### Test Data

- Samples: ~300,000 rows
- Size: 33 MB
- Features: 17

**Target Variable**: `diagnosed_diabetes` (Binary: 0 or 1)


## 👤 Author

**Nilesh Mishra**

- GitHub: [@nileshmishra](https://github.com/nileshmishra)
- LinkedIn: [Nilesh Mishra](https://linkedin.com/in/nileshmishra)
- Kaggle: [Your Kaggle Profile](https://www.kaggle.com/yourusername)

---

## 📝 License

This project is licensed under the MIT License.

**⭐ If you find this project helpful, please star the repository!**

**🏆 Competition**: [Playground Series S5E12](https://www.kaggle.com/competitions/playground-series-s5e12)

**📊 Score**: 0.70102 ROC-AUC
