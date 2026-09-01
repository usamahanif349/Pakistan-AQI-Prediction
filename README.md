# Pakistan-AQI-Prediction
ML/DL comparative study predicting Pakistan's AQI across 10 cities.
# Pakistan AQI Prediction: A Comparative Study of Machine Learning and Deep Learning Approaches

This project presents a comprehensive comparison of classical machine learning algorithms, an Artificial Neural Network (ANN), and a supplementary LSTM forecasting model for predicting Air Quality Index (AQI) values and health categories across 10 Pakistani cities using 10 years of monthly data (2015-2025).

## Key Results

| Model | Task | Score |
|---|---|---|
| Gradient Boosting (tuned) | Regression | R² = 0.974, MAE = 5.15 |
| Random Forest | Regression | R² = 0.973, MAE = 5.18 |
| Gradient Boosting | Classification | 91.7% accuracy |
| ANN | Regression | R² = 0.944 |
| ANN | Classification | 88.3% accuracy |
| LSTM (forecasting) | Time-series | R² = 0.883 |

**Finding:** Tree-based ensemble methods (Random Forest, XGBoost, Gradient Boosting) consistently outperformed both linear/kernel-based classical methods and a deep learning ANN approach, across regression and classification tasks — all results validated via 5-fold cross-validation.

## Project Structure
├── notebook/ # Full analysis notebook (data prep, EDA, modeling, evaluation)

├── data/ # Dataset (Pakistan AQI, 10 cities, 2015-2025) — sourced from Kaggle

├── paper/ # Full IEEE-format research paper (PDF + LaTeX source)

└── README.md


## Tech Stack
Python, Pandas, Scikit-learn, XGBoost, TensorFlow/Keras, Matplotlib, Seaborn

## Dataset
[Pakistan Air Quality Index: 10 Cities, 2015-2025](https://www.kaggle.com/datasets/alitaqishah/pakistan-air-quality-index-10-cities-20152025) (Kaggle)

## Paper
Full methodology, results, and discussion available in [`AQI_Prediction_Paper.pdf`](./AQI_Prediction_Paper.pdf).
