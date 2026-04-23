# ?? Thyroid Cancer Recurrence Predictor - Streamlit Deployment

A web application built on the best-performing **LightGBM** model (AUC = 0.983),
supporting both single-case and batch prediction.

---

## ?? File Structure

```
.
??? app.py                      # Streamlit main program
??? requirements.txt            # Python dependencies
??? run_app.bat                 # Windows one-click launcher
??? README_APP.md               # This document
??? modeldata_335_PTC.csv       # Original training data (335 PTC patients)
??? model-Copy3.ipynb           # Model training notebook
??? thyroid_cancer_9models_results_pureNoStacking_335PTCpatients_lightGBM/
    ??? BEST_MODEL_LightGBM.pkl     # Best model file
    ??? scaler.pkl                  # Standard scaler
    ??? model_ranking.csv           # Performance ranking of 9 models
    ??? testing_set.csv             # Test set (can be used as batch example)
    ??? *.png / *.pdf               # Training visualizations
```

---

## ?? Quick Start

### Option 1: Double-click (Windows, recommended)
```
Double-click run_app.bat
```
The script installs dependencies and starts the server; your browser will open
<http://localhost:8501> automatically.

### Option 2: Command line
```bash
# 1) Install dependencies (first time only)
pip install -r requirements.txt

# 2) Launch the app
streamlit run app.py
```

### Option 3: Custom port / external access
```bash
streamlit run app.py --server.port 8080 --server.address 0.0.0.0
```

---

## ??? Features

### 1. ?? Single Prediction
- Enter **16 clinical features** via an intuitive form
- Outputs: **recurrence probability, predicted class, 4-level risk tier, visualization**
- Includes clinical recommendation and medical disclaimer

### 2. ?? Batch Prediction
- Upload a CSV file (columns must match the template)
- Download the **CSV template** and **test-set example**
- View summary statistics and a probability histogram; export results as CSV

### 3. ?? Model Performance
- Full performance comparison of 9 models (AUC / F1 / Accuracy / Precision / Recall / Brier)
- Browse 11 visualizations generated during training (ROC, calibration, confusion matrix, SHAP, etc.)

### 4. ?? About
- Project background, dataset information, modeling workflow, feature descriptions

---

## ?? Input Features

| # | Feature | Type | Values |
|---|---------|------|--------|
| 1  | Age                   | Continuous       | 15-100 |
| 2  | Gender                | Binary           | 0=F, 1=M |
| 3  | Smoking               | Binary           | 0=No, 1=Yes |
| 4  | Hx Smoking            | Binary           | 0=No, 1=Yes |
| 5  | Hx Radiothreapy       | Binary           | 0=No, 1=Yes |
| 6  | Thyroid Function      | Multi-class (5)  | 0-4 |
| 7  | Physical Examination  | Multi-class (5)  | 0-4 |
| 8  | Adenopathy            | Multi-class (6)  | 0-5 |
| 9  | Pathology             | Binary           | 0=Micropapillary, 1=Papillary |
| 10 | Focality              | Binary           | 0=Multi, 1=Uni |
| 11 | Risk                  | Multi-class (3)  | 0=Low, 1=Intermediate, 2=High |
| 12 | T                     | Multi-class (7)  | 0-6 |
| 13 | N                     | Multi-class (3)  | 0=N0, 1=N1a, 2=N1b |
| 14 | M                     | Binary           | 0=M0, 1=M1 |
| 15 | Stage                 | Multi-class (5)  | 0=I, 1=II, 2=III, 3=IVA, 4=IVB |
| 16 | Response              | Multi-class (4)  | 0=Excellent, 1=Indeterminate, 2=Struct., 3=Biochem. |

> Batch CSVs should contain these **integer-encoded** values. The single-prediction
> form automatically maps human-readable options to integers.

---

## ?? Model Performance (test set, n = 101)

| Model | Accuracy | Precision | Recall | F1 | AUC | Brier |
|-------|----------|-----------|--------|-----|-----|-------|
| **LightGBM** ?      | 0.9703 | 1.0000 | 0.8889 | 0.9412 | **0.9830** | 0.0264 |
| CatBoost            | 0.9604 | 1.0000 | 0.8519 | 0.9200 | 0.9825 | 0.0382 |
| XGBoost             | 0.9703 | 1.0000 | 0.8889 | 0.9412 | 0.9785 | 0.0318 |
| Logistic Regression | 0.9208 | 0.8519 | 0.8519 | 0.8519 | 0.9730 | 0.0622 |
| SVM                 | 0.9010 | 0.8148 | 0.8148 | 0.8148 | 0.9690 | 0.0647 |
| KNN                 | 0.9307 | 1.0000 | 0.7407 | 0.8511 | 0.9632 | 0.0680 |
| Random Forest       | 0.9505 | 1.0000 | 0.8148 | 0.8980 | 0.9612 | 0.0469 |
| Naive Bayes         | 0.9010 | 0.9048 | 0.7037 | 0.7917 | 0.9577 | 0.0938 |
| Decision Tree       | 0.9307 | 0.9545 | 0.7778 | 0.8571 | 0.9497 | 0.0554 |

---

## ?? Troubleshooting

### Q1: Model file not found
Ensure this file exists:
```
thyroid_cancer_9models_results_pureNoStacking_335PTCpatients_lightGBM/BEST_MODEL_LightGBM.pkl
```
If missing, re-run `model-Copy3.ipynb` to retrain the models.

### Q2: Port already in use
```bash
streamlit run app.py --server.port 8502
```

### Q3: How to deploy remotely?

#### Streamlit Community Cloud (推荐免费部署)
1. 将代码推送到 GitHub 仓库
2. 访问 <https://share.streamlit.io> 并连接你的 GitHub 账号
3. 选择仓库和 `app.py` 文件，点击 Deploy
4. 确保模型目录 `thyroid_cancer_9models_results_pureNoStacking_335PTCpatients_lightGBM/` 已上传到 GitHub

#### Docker 部署
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

构建并运行：
```bash
docker build -t ptc-predictor .
docker run -p 8501:8501 ptc-predictor
```

#### 云服务器部署
```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

---

## ?? Disclaimer

This system is intended for **research and educational purposes only**.
Predictions must not replace professional medical diagnosis.
Any medical decision must be made by a qualified physician based on a complete clinical evaluation.
