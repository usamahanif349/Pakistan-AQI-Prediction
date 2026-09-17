# Pakistan AQI Prediction: A Comparative Study of Machine Learning and Deep Learning Approaches

This project presents a comprehensive comparison of classical machine learning algorithms, an Artificial Neural Network (ANN), and a supplementary LSTM forecasting model for predicting Air Quality Index (AQI) values and health categories across 10 Pakistani cities using 10 years of monthly data (2015-2025).

## 🔴 Live App

**[Try the interactive AQI Predictor →](https://pakistan-aqi-prediction-frhdsrc2ivtubpkdotcau5.streamlit.app/)**

Explore predictions, an all-cities map, historical trends, city comparisons, and a seasonal heatmap — all powered by the tuned Gradient Boosting model from this study.

## Key Results

| Model | Task | Score |
|---|---|---|
| Gradient Boosting (tuned) | Regression | R² = 0.974, MAE = 5.15 |
| Random Forest | Regression | R² = 0.973, MAE = 5.18 |
| Gradient Boosting | Classification | 91.7% accuracy |
| Stacking Ensemble (RF + XGBoost + GB) | Both | R² = 0.974, 90.9% accuracy |
| ANN | Regression | R² = 0.944 |
| ANN | Classification | 88.3% accuracy |
| LSTM (forecasting) | Time-series | R² = 0.883 |
| LSTM-XGBoost Hybrid | Forecasting | R² = 0.947 |

**Finding:** Tree-based ensemble methods (Random Forest, XGBoost, Gradient Boosting) consistently outperformed both linear/kernel-based classical methods and deep learning approaches (ANN, LSTM), across regression and classification tasks — all results validated via 5-fold cross-validation. SHAP analysis confirmed smog season and city identity as the dominant predictors of AQI.

## Project Structure
```
├── Pakistan_AQI_Prediction_ML_ANN_Comparison.ipynb   # Full analysis notebook
├── Pakistan_AQI_Prediction_Paper.pdf                 # IEEE-format research paper
├── app.py                                            # Live AQI predictor (Streamlit)
├── aqi_model.pkl, aqi_classifier.pkl                 # Trained models (regression + classification)
├── le_city.pkl, le_season.pkl, le_province.pkl       # Encoders used by the app
├── city_lookup.json, city_history.csv                # Supporting data for the app
├── *.csv                                             # Source dataset (Kaggle)
├── *.png                                             # Result charts (comparisons, SHAP, hybrid boxplot)
└── README.md
```

## Tech Stack
Python, Pandas, Scikit-learn, XGBoost, TensorFlow/Keras, Matplotlib, Seaborn, Streamlit, SHAP

## Dataset
[Pakistan Air Quality Index: 10 Cities, 2015-2025](https://www.kaggle.com/datasets/alitaqishah/pakistan-air-quality-index-10-cities-20152025) (Kaggle)

## Paper
Full methodology, results, and discussion available in [`Pakistan_AQI_Prediction_Paper.pdf`](./Pakistan_AQI_Prediction_Paper.pdf).
