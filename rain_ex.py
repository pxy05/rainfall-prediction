# ============================================
# CM22009 - Rainfall Prediction using KNN
# ============================================

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Load rainfall data ---
def load_rainfall(station_file):
    """Each file is a numpy array with columns: Year, Month, MinTemp, MaxTemp, FrostDays, MonthlyRainfall"""
    arr = np.load(station_file)
    df = pd.DataFrame(arr, columns=['Year','Month','MinTemp','MaxTemp','FrostDays','MonthlyRainfall'])
    df['Station'] = station_file.replace('.npy','')
    return df

def load_rainfall_all():
    dfs = []
    for name in ['Cardiff.npy','Aberporth.npy','Valley.npy']:
        dfs.append(load_rainfall(name))
    return pd.concat(dfs, ignore_index=True)

# Load and prepare dataset
df = load_rainfall_all()
df = df.sort_values(['Year', 'Month']).reset_index(drop=True)
print("Dataset shape:", df.shape)
print(df.head())

import math

# --- Train/test split to avoid temporal leakage ---
split_index = int(len(df) * 0.8)
y = df['MonthlyRainfall']
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

feature_sets = {
    'baseline': ['MinTemp', 'MaxTemp', 'FrostDays'],
    'seasonal': ['MinTemp', 'MaxTemp', 'FrostDays', 'Month']
}

# --- Hyperparameter tuning ---
param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9, 11, 15],
    'knn__weights': ['uniform', 'distance'],
    'knn__p': [1, 2]  # 1 = Manhattan, 2 = Euclidean
}

model_evaluations = {}
results_summary = []

for name, columns in feature_sets.items():
    X = df[columns]
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsRegressor())
    ])

    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n=== KNN ({name}) ===")
    print("Best Parameters:", grid.best_params_)
    print("Best CV RMSE:", -grid.best_score_)
    print("Test Performance:")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"R²: {r2:.3f}")

    model_evaluations[name] = {
        'model': best_model,
        'X_train': X_train,
        'X_test': X_test,
        'y_pred': y_pred,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

    label = "KNN (baseline features)" if name == 'baseline' else "KNN (+ Month feature)"
    results_summary.append({
        'Model': label,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    })

# --- Diagnostic plots for A3 poster ---

# 1. Predicted vs Actual
for name, evaluation in model_evaluations.items():
    y_pred = evaluation['y_pred']
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    title_suffix = " (+ Month)" if name == 'seasonal' else " (Baseline)"
    plt.xlabel("Actual Monthly Rainfall (mm)")
    plt.ylabel("Predicted Monthly Rainfall (mm)")
    plt.title(f"KNN: Predicted vs Actual{title_suffix}")
    plt.tight_layout()
    plt.show()

# 2. Residual Distribution
for name, evaluation in model_evaluations.items():
    residuals = y_test - evaluation['y_pred']
    plt.figure(figsize=(6,4))
    sns.histplot(residuals, kde=True)
    title_suffix = " (+ Month)" if name == 'seasonal' else " (Baseline)"
    plt.title(f"KNN Residual Distribution{title_suffix}")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.tight_layout()
    plt.show()

# 3. RMSE vs k plot (optional, nice visual for poster)
k_values = [3,5,7,9,11,15]
for name, evaluation in model_evaluations.items():
    rmse_scores = []
    for k in k_values:
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsRegressor(n_neighbors=k))
        ])
        model.fit(evaluation['X_train'], y_train)
        y_pred_k = model.predict(evaluation['X_test'])
        rmse_k = math.sqrt(mean_squared_error(y_test, y_pred_k))
        rmse_scores.append(rmse_k)

    plt.figure(figsize=(6,4))
    plt.plot(k_values, rmse_scores, marker='o')
    title_suffix = " (+ Month)" if name == 'seasonal' else " (Baseline)"
    plt.title(f"KNN: RMSE vs k{title_suffix}")
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.show()

# --- Save results for comparison with Random Forest later ---
results_knn = pd.DataFrame(results_summary)
results_knn.to_csv("knn_results.csv", index=False)
print("\nSaved knn_results.csv")
