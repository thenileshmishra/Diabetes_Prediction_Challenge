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
- **Dataset**: ~400,000 training samples, ~260,000 test samples
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

This isn't just a competition submission—it's a **production-ready ML system** with:

### Machine Learning Pipeline

- Data ingestion with quality checks
- Advanced feature engineering (cardiovascular, lipid profile, lifestyle metrics)
- Multi-model training with cross-validation
- Ensemble prediction with weighted averaging
- Comprehensive evaluation metrics

### FastAPI Backend

- RESTful API with automatic documentation
- Real-time predictions via HTTP endpoints
- Health monitoring and status checks
- Type-safe request/response validation (Pydantic)
- CORS enabled for frontend integration

### Web Interface

- Professional HTML/CSS/JS frontend
- Responsive design (mobile-friendly)
- Interactive prediction form with 17 input fields
- Visual results display with risk categorization
- Real-time validation and error handling

### Deployment Ready

- Docker containerization
- Docker Compose orchestration
- AWS EC2 deployment capability
- Health checks and monitoring
- Production-ready configuration

---

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
├── app/                   # FastAPI Backend
│   ├── main.py           # API routes & endpoints
│   ├── schemas.py        # Request/response models
│   ├── predict.py        # ML prediction engine
│   └── health.py         # Health checks
│
├── static/               # Web Frontend
│   ├── index.html       # Prediction interface
│   ├── style.css        # Responsive styling
│   └── script.js        # API communication
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

---

## 🚀 Quick Start

### Option 1: Local Setup

```bash
# 1. Clone repository
git clone <your-repo-url>
cd Kaggel

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train models (or use pre-trained)
python src/main.py --ingest --train

# 4. Start API server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 5. Open browser → http://localhost:8000
```

### Option 2: Docker (Recommended)

```bash
# 1. Build and start
docker-compose up -d

# 2. Check status
docker-compose ps

# 3. Open browser → http://localhost:8000
```

---

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

## 📡 API Usage

### Endpoints

| Endpoint | Method | Description |
| --- | --- | --- |
| `/` | GET | Web interface |
| `/health` | GET | Health check & model status |
| `/predict` | POST | Make diabetes prediction |
| `/info` | GET | API & model information |
| `/docs` | GET | Interactive API documentation |

### Example: Make a Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 52.0,
    "gender": 1.0,
    "bmi": 29.3,
    "waist_to_hip_ratio": 0.94,
    "systolic_bp": 138.0,
    "diastolic_bp": 88.0,
    "heart_rate": 75.0,
    "cholesterol": 215.0,
    "ldl": 135.0,
    "hdl": 45.0,
    "triglycerides": 175.0,
    "physical_activity": 2.5,
    "screen_time": 5.0,
    "sleep_duration": 6.5,
    "hypertension_history": 1.0,
    "cardiovascular_history": 0.0,
    "family_history": 1.0
  }'
```

**Response:**

```json
{
  "prediction": 1,
  "probability": 0.7234,
  "risk_level": "High"
}
```

---

## 🐳 Docker Deployment

### Quick Start

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### What's Included

- ✅ Python 3.10 runtime
- ✅ All ML dependencies (LightGBM, XGBoost, CatBoost)
- ✅ FastAPI with Uvicorn server
- ✅ Trained models (15 model files)
- ✅ Web interface (HTML/CSS/JS)
- ✅ Health checks and monitoring
- ✅ Resource limits (CPU, memory)

### Image Details

- **Base**: python:3.10-slim
- **Size**: ~400MB (optimized)
- **Port**: 8000
- **Health Check**: Every 30s

---

## 🛠️ Tech Stack

### Machine Learning

- Python 3.10+
- LightGBM, XGBoost, CatBoost
- Scikit-learn
- NumPy, Pandas

### Backend

- FastAPI
- Uvicorn (ASGI server)
- Pydantic (validation)

### Frontend

- HTML5, CSS3, JavaScript
- Responsive design
- Fetch API

### DevOps

- Docker
- Docker Compose
- AWS EC2 ready

---

## 📊 Dataset Information

**Source**: [Kaggle Playground Series S5E12](https://www.kaggle.com/competitions/playground-series-s5e12)

### Training Data

- Samples: ~400,000 rows
- Size: 79 MB
- Features: 17 + target
- Class Distribution: Imbalanced

### Test Data

- Samples: ~260,000 rows
- Size: 33 MB
- Features: 17

**Target Variable**: `diagnosed_diabetes` (Binary: 0 or 1)

---

## 🧪 Testing

```bash
# Run automated API tests
python test_api.py

# Test health endpoint
curl http://localhost:8000/health

# Interactive testing
# Open: http://localhost:8000/docs
```

---

## 📚 Documentation

- **Quick Start**: See above sections
- **API Documentation**: <http://localhost:8000/docs> (auto-generated)
- **Code Comments**: Inline documentation throughout
- **Docker Guide**: `docker-compose up -d` to start

---

## 🎯 Use Cases

### 1. Healthcare Screening

Preliminary diabetes risk assessment based on patient vitals and lifestyle

### 2. Research & Education

Demonstrating ML ensemble methods and feature engineering

### 3. API Integration

RESTful API for integration with existing health systems

### 4. Portfolio Project

Production-ready ML deployment showcasing full-stack skills

---

## 🔮 Future Enhancements

- [ ] SHAP values for prediction explainability
- [ ] User authentication (JWT)
- [ ] Database integration for prediction history
- [ ] Model monitoring and drift detection
- [ ] A/B testing framework
- [ ] Hyperparameter optimization (Optuna)
- [ ] CI/CD pipeline (GitHub Actions)

---

## 👤 Author

**Nilesh Mishra**

- GitHub: [@nileshmishra](https://github.com/nileshmishra)
- LinkedIn: [Nilesh Mishra](https://linkedin.com/in/nileshmishra)
- Kaggle: [Your Kaggle Profile](https://www.kaggle.com/yourusername)

---

## 📝 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **Kaggle** - For hosting the Playground Series competition
- **FastAPI** - Excellent web framework
- **LightGBM, XGBoost, CatBoost** - Powerful ML libraries
- **Docker** - Containerization platform

---

## 📞 Support

For questions or issues:

1. Check the [API Documentation](<http://localhost:8000/docs>)
2. Review code comments
3. Open an issue on GitHub
4. Test with the web interface at <http://localhost:8000>

---

**⭐ If you find this project helpful, please star the repository!**

**🏆 Competition**: [Playground Series S5E12](https://www.kaggle.com/competitions/playground-series-s5e12)

**📊 Score**: 0.70102 ROC-AUC
