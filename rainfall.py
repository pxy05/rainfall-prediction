# Load Dataset and split into features and actual rainfall

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

columns = ['year', 'month', 'max_temp', 'min_temp', 'frost_days', 'monthly_rainfall']   

cardiff_dframe = pd.DataFrame(np.load('cardiff.npy'), columns=columns)
aberporth_dframe = pd.DataFrame(np.load('aberporth.npy'), columns=columns)
valley_dframe = pd.DataFrame(np.load('valley.npy'), columns=columns)

dataset = (
    pd.concat([cardiff_dframe, aberporth_dframe, valley_dframe], ignore_index=True)
    .sort_values(['year', 'month'])
    .reset_index(drop=True)
)

month_encoder = OneHotEncoder(sparse_output=False)
month_encoded = month_encoder.fit_transform(dataset[['month']])
month_encoded_dataset = pd.DataFrame(month_encoded, columns=month_encoder.get_feature_names_out(['month']))

features = pd.concat([dataset[['min_temp', 'max_temp', 'frost_days']], month_encoded_dataset], axis=1)
features_wo_month = dataset[['min_temp', 'max_temp', 'frost_days']]
actual_rainfall = dataset['monthly_rainfall']

# ------------------------------------------------------------

# Split into training and testing sets

from sklearn.preprocessing import StandardScaler

split_idx = int(len(features) * 0.8)

train_features = features.iloc[:split_idx]
test_features = features.iloc[split_idx:]
train_actual = actual_rainfall.iloc[:split_idx]
test_actual = actual_rainfall.iloc[split_idx:]

scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
test_features_scaled = scaler.transform(test_features)

train_features_wo_month = features_wo_month.iloc[:split_idx]
test_features_wo_month = features_wo_month.iloc[split_idx:]
train_actual_wo_month = actual_rainfall.iloc[:split_idx]
test_actual_wo_month = actual_rainfall.iloc[split_idx:]

train_features_scaled_wo_month = scaler.fit_transform(train_features_wo_month)
test_features_scaled_wo_month = scaler.transform(test_features_wo_month)

# ------------------------------------------------------------

# Plot data

import seaborn as sns
import matplotlib.pyplot as plt

# sns.pairplot(dataset[['min_temp', 'max_temp', 'frost_days', 'monthly_rainfall']])
# sns.heatmap(dataset.corr(), annot=True)
# plt.show()
# low_rainfall_days = (dataset['monthly_rainfall'] <= 50).sum()
# medium_rainfall_days = ((dataset['monthly_rainfall'] > 50) & (dataset['monthly_rainfall'] <= 100)).sum()
# high_rainfall_days = (dataset['monthly_rainfall'] > 100).sum()

# print("Number of days with low rainfall:", low_rainfall_days)
# print("Number of days with medium rainfall:", medium_rainfall_days)
# print("Number of days with high rainfall:", high_rainfall_days)

# print(train_features_scaled[0:12])

# ------------------------------------------------------------

# Train decision tree

from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(random_state=69)
tree.fit(train_features_scaled, train_actual)

tree_wo_month = DecisionTreeRegressor(random_state=69)
tree_wo_month.fit(train_features_scaled_wo_month, train_actual_wo_month)

# ------------------------------------------------------------

# Find optimal number of estimators
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

potential_est_vals = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

# Separate metrics for rf (with month) and rf_wo_month (without month)
mse_with_month, mae_with_month, r2_with_month = [], [], []
mse_wo_month, mae_wo_month, r2_wo_month = [], [], []
OOB_scores_with_month, OOB_scores_wo_month = [], []

rf = RandomForestRegressor(n_estimators=potential_est_vals[0], warm_start=True,
                           oob_score=True, n_jobs=-1, random_state=69)
rf.fit(train_features_scaled, train_actual)

rf_wo_month = RandomForestRegressor(n_estimators=potential_est_vals[0], warm_start=True,
                                    oob_score=True, n_jobs=-1, random_state=69)
rf_wo_month.fit(train_features_scaled_wo_month, train_actual_wo_month)

# Collect metrics for the initial (first) n_estimators value
OOB_scores_with_month.append(rf.oob_score_)
OOB_scores_wo_month.append(rf_wo_month.oob_score_)

mse_with_month.append(mean_squared_error(test_actual, rf.predict(test_features_scaled)))
mse_wo_month.append(mean_squared_error(test_actual, rf_wo_month.predict(test_features_scaled_wo_month)))
mae_with_month.append(mean_absolute_error(test_actual, rf.predict(test_features_scaled)))
mae_wo_month.append(mean_absolute_error(test_actual, rf_wo_month.predict(test_features_scaled_wo_month)))
r2_with_month.append(r2_score(test_actual, rf.predict(test_features_scaled)))
r2_wo_month.append(r2_score(test_actual, rf_wo_month.predict(test_features_scaled_wo_month)))

for est in potential_est_vals[1:]:
    rf.set_params(n_estimators=est)
    rf.fit(train_features_scaled, train_actual)   # warm_start keeps prior trees, adds the difference

    rf_wo_month.set_params(n_estimators=est)
    rf_wo_month.fit(train_features_scaled_wo_month, train_actual_wo_month)

    OOB_scores_with_month.append(rf.oob_score_)
    OOB_scores_wo_month.append(rf_wo_month.oob_score_)

    mse_with_month.append(mean_squared_error(test_actual, rf.predict(test_features_scaled)))
    mse_wo_month.append(mean_squared_error(test_actual, rf_wo_month.predict(test_features_scaled_wo_month)))
    mae_with_month.append(mean_absolute_error(test_actual, rf.predict(test_features_scaled)))
    mae_wo_month.append(mean_absolute_error(test_actual, rf_wo_month.predict(test_features_scaled_wo_month)))
    r2_with_month.append(r2_score(test_actual, rf.predict(test_features_scaled)))
    r2_wo_month.append(r2_score(test_actual, rf_wo_month.predict(test_features_scaled_wo_month)))

# Plot MSE
plt.figure(figsize=(10, 5))
plt.plot(potential_est_vals, mse_with_month, label="MSE (with month)", marker='o', color='b')
plt.plot(potential_est_vals, mse_wo_month, label="MSE (without month)", marker='o', linestyle='dashed', color='b')
plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("Mean Squared Error")
plt.title("MSE vs Number of Trees")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot MAE
plt.figure(figsize=(10, 5))
plt.plot(potential_est_vals, mae_with_month, label="MAE (with month)", marker='o', color='g')
plt.plot(potential_est_vals, mae_wo_month, label="MAE (without month)", marker='o', linestyle='dashed', color='g')
plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("Mean Absolute Error")
plt.title("MAE vs Number of Trees")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot R^2
plt.figure(figsize=(10, 5))
plt.plot(potential_est_vals, r2_with_month, label="R2 (with month)", marker='o', color='r')
plt.plot(potential_est_vals, r2_wo_month, label="R2 (without month)", marker='o', linestyle='dashed', color='r')
plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("R2 Score")
plt.title("R2 vs Number of Trees")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot OOB Score
plt.figure(figsize=(10, 5))
plt.plot(potential_est_vals, OOB_scores_with_month, label="OOB Score (with month)", marker='o', color='purple')
plt.plot(potential_est_vals, OOB_scores_wo_month, label="OOB Score (without month)", marker='o', linestyle='dashed', color='purple')
plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("OOB Score")
plt.title("OOB Score vs Number of Trees")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

''' MENTION N_EST IS LOW DUE TO THE LOW NUMBER OF ACTUAL DATA POINTS '''
''' OOB SCORE IS AN EXTRA METRIC TO EVALUATE THE MODEL '''

randf = RandomForestRegressor(n_estimators=100, random_state=69, oob_score=True)
randf.fit(train_features_scaled, train_actual)

randf_wo_month = RandomForestRegressor(n_estimators=100, random_state=69, oob_score=True)
randf_wo_month.fit(train_features_scaled_wo_month, train_actual_wo_month)

# ------------------------------------------------------------

# Evaluate models

import numpy as np

tree_preds = tree.predict(test_features_scaled)
rand_forest_preds = randf.predict(test_features_scaled)

tree_wo_month_preds = tree_wo_month.predict(test_features_scaled_wo_month)
rand_forest_wo_month_preds = randf_wo_month.predict(test_features_scaled_wo_month)

dec_place = 4

def eval_stats(model, test_features, test_actuals):
    predictions = model.predict(test_features)
    measurements = {
        'MAE': round(mean_absolute_error(test_actuals, predictions), dec_place),
        'RMSE': round(float(np.sqrt(mean_squared_error(test_actuals, predictions))), dec_place),
        'R2': round(r2_score(test_actuals, predictions), dec_place),
    }

    if hasattr(model, 'oob_score_'):
        measurements['OOB Score'] = round(model.oob_score_, dec_place)
    
    return measurements

tree_results = eval_stats(tree, test_features_scaled, test_actual)
tree_wo_month_results = eval_stats(tree_wo_month, test_features_scaled_wo_month, test_actual_wo_month)

rand_forest_results = eval_stats(randf, test_features_scaled, test_actual)
rand_forest_wo_month_results = eval_stats(randf_wo_month, test_features_scaled_wo_month, test_actual_wo_month)

print("Tree results:")
print("   ", tree_results)
print("Tree results without month:")
print("   ", tree_wo_month_results)

print("Random forest results:")
print("   ", rand_forest_results)
print("Random forest results without month:")
print("   ", rand_forest_wo_month_results)

