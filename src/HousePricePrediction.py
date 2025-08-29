import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn  # Added to fix NameError
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, StandardScaler, FunctionTransformer, \
    PolynomialFeatures
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score  # Added r2_score
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import joblib
import sys
from tabulate import tabulate
from scipy.stats import linregress

# Debug: Print Python and library versions
print(f"Python version: {sys.version}")
print(f"pandas version: {pd.__version__}")
print(f"scikit-learn version: {sklearn.__version__}")
print(f"xgboost version: {xgb.__version__}")
print(f"lightgbm version: {lgb.__version__}")
print(f"catboost version: {CatBoostRegressor.__module__}")
print(f"joblib version: {joblib.__version__}")

# Load dataset
try:
    strat_train_set = pd.read_csv(r"C:\Users\Momo\Downloads\housing.csv")
    print("Dataset loaded successfully.")
    print(f"Dataset columns: {strat_train_set.columns.tolist()}")
    print(f"Dataset shape: {strat_train_set.shape}")
    print(f"Missing values in dataset:\n{strat_train_set.isnull().sum()}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Preparing the Data
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# Split data into train and test sets
housing_train, housing_test, labels_train, labels_test = train_test_split(
    housing, housing_labels, test_size=0.2, random_state=42
)
print(f"Train set size: {len(housing_train)}, Test set size: {len(housing_test)}")
print(f"Train data columns: {housing_train.columns.tolist()}")

# Separate numerical and categorical features
housing_num = housing.drop("ocean_proximity", axis=1)
housing_cat = housing[["ocean_proximity"]]

# Custom Transformers
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]

log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
rbf_transformer = FunctionTransformer(rbf_kernel, kw_args=dict(Y=[[35.]], gamma=0.1))
sf_coords = [37.7749, -122.41]
sf_transformer = FunctionTransformer(rbf_kernel, kw_args=dict(Y=[sf_coords], gamma=0.1))

# Transformation Pipelines
num_pipeline = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler())  # Changed to "mean"
cat_pipeline = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="mean"),  # Changed to "mean"
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler()
    )

log_pipeline = make_pipeline(
    SimpleImputer(strategy="mean"),  # Changed to "mean"
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler()
)

default_num_pipeline = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler())  # Changed to "mean"

# Interaction Features Pipeline
interaction_pipeline = make_pipeline(
    SimpleImputer(strategy="mean"),
    PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
    # Add interactions, e.g., total_rooms * median_income
    StandardScaler()
)

# ColumnTransformer for preprocessing
preprocessing = ColumnTransformer([
    ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
    ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
    ("people_per_house", ratio_pipeline(), ["population", "households"]),
    ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population", "households", "median_income"]),
    ("interactions", interaction_pipeline, ["total_rooms", "median_income"]),  # Added interactions
    ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
], remainder=default_num_pipeline)

# Prepare data (fit on train)
try:
    preprocessing.fit(housing_train)
    print("Preprocessing fitted successfully.")
    print("Prepared Data Shape:", preprocessing.transform(housing_train).shape)
    print("Feature Names:", preprocessing.get_feature_names_out())
except Exception as e:
    print(f"Error in preprocessing: {e}")
    exit(1)

# Save preprocessing pipeline
print("Attempting to save preprocessing pipeline...")
try:
    joblib.dump(preprocessing, r'C:\Users\Momo\PycharmProjects\PythonProject2\preprocessing.pkl')
    print("Preprocessing pipeline saved as 'C:\\Users\\Momo\\PycharmProjects\\PythonProject2\\preprocessing.pkl'")
except Exception as e:
    print(f"Error saving preprocessing pipeline: {e}")

# Train Linear Regression Model
lin_reg = make_pipeline(preprocessing, LinearRegression())
try:
    lin_reg.fit(housing_train, labels_train)
    print("Linear Regression trained successfully.")
except Exception as e:
    print(f"Error training Linear Regression: {e}")
    exit(1)

# Save Linear Regression model
try:
    joblib.dump(lin_reg, r'C:\Users\Momo\PycharmProjects\PythonProject2\linear_regression_model.pkl')
    print("Linear Regression model saved as 'C:\\Users\\Momo\\PycharmProjects\\PythonProject2\\linear_regression_model.pkl'")
except Exception as e:
    print(f"Error saving Linear Regression model: {e}")
    exit(1)

# Predictions and Evaluation for Linear Regression
try:
    train_predictions = lin_reg.predict(housing_train)
    train_mse = mean_squared_error(labels_train, train_predictions)
    train_r2 = r2_score(labels_train, train_predictions)
    print(f"Linear Regression Train RMSE: {np.sqrt(train_mse):.2f}")
    print(f"Linear Regression Train R² Score: {train_r2:.4f}")

    test_predictions = lin_reg.predict(housing_test)
    test_mse = mean_squared_error(labels_test, test_predictions)
    test_r2 = r2_score(labels_test, test_predictions)
    print(f"Linear Regression Test RMSE: {np.sqrt(test_mse):.2f}")
    print(f"Linear Regression Test R² Score: {test_r2:.4f}")

    print("Linear Regression Predictions (first 5, rounded):", test_predictions[:5].round(-2))
    print("True Test Labels (first 5):", labels_test.iloc[:5].values)
except Exception as e:
    print(f"Error evaluating Linear Regression: {e}")
    exit(1)

# Cross-Validation for Linear Regression
try:
    lin_reg_scores = cross_val_score(lin_reg, housing_train, labels_train, scoring="neg_root_mean_squared_error", cv=5)
    print(f"Linear Regression CV RMSE: {-lin_reg_scores.mean():.2f}")
except Exception as e:
    print(f"Error in Linear Regression CV: {e}")
    exit(1)

# Hyperparameter Tuning for Random Forest with RandomizedSearchCV
forest_pipeline = make_pipeline(
    preprocessing,
    RandomForestRegressor(random_state=33)
)

param_dist_rf = {
    'randomforestregressor__n_estimators': [100, 200, 300],
    'randomforestregressor__max_depth': [5, 10, 15, 20],
    'randomforestregressor__min_samples_split': [2, 5, 10, 15],
    'randomforestregressor__min_samples_leaf': [1, 2, 4],
    'randomforestregressor__max_features': ['sqrt', 'log2', None]
}

try:
    print("Starting Random Forest RandomizedSearchCV...")
    random_search_rf = RandomizedSearchCV(
        forest_pipeline,
        param_distributions=param_dist_rf,
        n_iter=30,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=2,
        random_state=33
    )
    random_search_rf.fit(housing_train, labels_train)
    print("Random Forest RandomizedSearchCV completed.")
    print("Best Parameters (RF):", random_search_rf.best_params_)
    print(f"Tuned Random Forest CV RMSE: {-random_search_rf.best_score_:.2f}")

    best_forest = random_search_rf.best_estimator_
    best_forest_train_pred = best_forest.predict(housing_train)
    train_mse_rf = mean_squared_error(labels_train, best_forest_train_pred)
    train_r2_rf = r2_score(labels_train, best_forest_train_pred)
    print(f"Tuned Random Forest Train RMSE: {np.sqrt(train_mse_rf):.2f}")
    print(f"Tuned Random Forest Train R² Score: {train_r2_rf:.4f}")

    best_forest_test_pred = best_forest.predict(housing_test)
    test_mse_rf = mean_squared_error(labels_test, best_forest_test_pred)
    test_r2_rf = r2_score(labels_test, best_forest_test_pred)
    print(f"Tuned Random Forest Test RMSE: {np.sqrt(test_mse_rf):.2f}")
    print(f"Tuned Random Forest Test R² Score: {test_r2_rf:.4f}")

    # Feature Importance for Random Forest
    best_rf = best_forest.named_steps['randomforestregressor']
    feature_importance_rf = pd.Series(best_rf.feature_importances_, index=preprocessing.get_feature_names_out())

    # Save Random Forest model
    joblib.dump(best_forest, 'random_forest_model.pkl')
    print("Random Forest model saved as 'random_forest_model.pkl'")
except Exception as e:
    print(f"Error in Random Forest RandomizedSearchCV: {e}")
    exit(1)

# XGBoost Model with Tuning
xgb_pipeline = make_pipeline(
    preprocessing,
    xgb.XGBRegressor(random_state=42, enable_categorical=True)
)

param_dist_xgb = {
    'xgbregressor__n_estimators': [100, 200, 300],
    'xgbregressor__max_depth': [3, 5, 7, 10],
    'xgbregressor__learning_rate': [0.01, 0.05, 0.1, 0.3],
    'xgbregressor__reg_alpha': [0, 0.1, 1],
    'xgbregressor__reg_lambda': [0, 1, 10]
}

try:
    print("Starting XGBoost RandomizedSearchCV...")
    random_search_xgb = RandomizedSearchCV(
        xgb_pipeline,
        param_distributions=param_dist_xgb,
        n_iter=30,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=2,
        random_state=42,
        error_score='raise'
    )
    random_search_xgb.fit(housing_train, labels_train)
    print("XGBoost RandomizedSearchCV completed.")
    print("Best Parameters (XGBoost):", random_search_xgb.best_params_)
    print(f"Tuned XGBoost CV RMSE: {-random_search_xgb.best_score_:.2f}")

    best_xgb = random_search_xgb.best_estimator_
    best_xgb_train_pred = best_xgb.predict(housing_train)
    train_mse_xgb = mean_squared_error(labels_train, best_xgb_train_pred)
    train_r2_xgb = r2_score(labels_train, best_xgb_train_pred)
    print(f"Tuned XGBoost Train RMSE: {np.sqrt(train_mse_xgb):.2f}")
    print(f"Tuned XGBoost Train R² Score: {train_r2_xgb:.4f}")

    best_xgb_test_pred = best_xgb.predict(housing_test)
    test_mse_xgb = mean_squared_error(labels_test, best_xgb_test_pred)
    test_r2_xgb = r2_score(labels_test, best_xgb_test_pred)
    print(f"Tuned XGBoost Test RMSE: {np.sqrt(test_mse_xgb):.2f}")
    print(f"Tuned XGBoost Test R² Score: {test_r2_xgb:.4f}")

    # Feature Importance for XGBoost
    best_xgb_model = best_xgb.named_steps['xgbregressor']
    feature_importance_xgb = pd.Series(best_xgb_model.feature_importances_, index=preprocessing.get_feature_names_out())

    # Save XGBoost model
    joblib.dump(best_xgb, 'xgboost_model.pkl')
    print("XGBoost model saved as 'xgboost_model.pkl'")
except Exception as e:
    print(f"Error in XGBoost RandomizedSearchCV: {e}")
    exit(1)

# CatBoost Model
catboost_pipeline = make_pipeline(
    preprocessing,
    CatBoostRegressor(random_state=42, verbose=0)
)

param_dist_cat = {
    'catboostregressor__iterations': [100, 200, 300],
    'catboostregressor__depth': [3, 5, 7],
    'catboostregressor__learning_rate': [0.01, 0.1, 0.3],
    'catboostregressor__l2_leaf_reg': [1, 3, 5],
    'catboostregressor__bagging_temperature': [0, 1, 2]
}

try:
    print("Starting CatBoost RandomizedSearchCV...")
    random_search_cat = RandomizedSearchCV(
        catboost_pipeline,
        param_distributions=param_dist_cat,
        n_iter=15,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    random_search_cat.fit(housing_train, labels_train)
    print("CatBoost RandomizedSearchCV completed.")
    print("Best Parameters (CatBoost):", random_search_cat.best_params_)
    print(f"Tuned CatBoost CV RMSE: {-random_search_cat.best_score_:.2f}")

    best_cat = random_search_cat.best_estimator_
    best_cat_train_pred = best_cat.predict(housing_train)
    train_mse_cat = mean_squared_error(labels_train, best_cat_train_pred)
    train_r2_cat = r2_score(labels_train, best_cat_train_pred)
    print(f"Tuned CatBoost Train RMSE: {np.sqrt(train_mse_cat):.2f}")
    print(f"Tuned CatBoost Train R² Score: {train_r2_cat:.4f}")

    best_cat_test_pred = best_cat.predict(housing_test)
    test_mse_cat = mean_squared_error(labels_test, best_cat_test_pred)
    test_r2_cat = r2_score(labels_test, best_cat_test_pred)
    print(f"Tuned CatBoost Test RMSE: {np.sqrt(test_mse_cat):.2f}")
    print(f"Tuned CatBoost Test R² Score: {test_r2_cat:.4f}")

    # Save CatBoost model
    joblib.dump(best_cat, 'catboost_model.pkl')
    print("CatBoost model saved as 'catboost_model.pkl'")
except Exception as e:
    print(f"Error in CatBoost RandomizedSearchCV: {e}")

# LightGBM Model
lgb_pipeline = make_pipeline(
    preprocessing,
    lgb.LGBMRegressor(random_state=42, verbose=-1)
)

param_dist_lgb = {
    'lgbmregressor__n_estimators': [100, 200, 300],
    'lgbmregressor__max_depth': [3, 5, 7],
    'lgbmregressor__learning_rate': [0.01, 0.1, 0.3],
    'lgbmregressor__reg_alpha': [0, 0.1, 1],
    'lgbmregressor__reg_lambda': [0, 1, 10]
}

try:
    print("Starting LightGBM RandomizedSearchCV...")
    random_search_lgb = RandomizedSearchCV(
        lgb_pipeline,
        param_distributions=param_dist_lgb,
        n_iter=15,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=2,
        random_state=42,
        error_score='raise'
    )
    random_search_lgb.fit(housing_train, labels_train)
    print("LightGBM RandomizedSearchCV completed.")
    print("Best Parameters (LightGBM):", random_search_lgb.best_params_)
    print(f"Tuned LightGBM CV RMSE: {-random_search_lgb.best_score_:.2f}")

    best_lgb = random_search_lgb.best_estimator_
    best_lgb_train_pred = best_lgb.predict(housing_train)
    train_mse_lgb = mean_squared_error(labels_train, best_lgb_train_pred)
    train_r2_lgb = r2_score(labels_train, best_lgb_train_pred)
    print(f"Tuned LightGBM Train RMSE: {np.sqrt(train_mse_lgb):.2f}")
    print(f"Tuned LightGBM Train R² Score: {train_r2_lgb:.4f}")

    best_lgb_test_pred = best_lgb.predict(housing_test)
    test_mse_lgb = mean_squared_error(labels_test, best_lgb_test_pred)
    test_r2_lgb = r2_score(labels_test, best_lgb_test_pred)
    print(f"Tuned LightGBM Test RMSE: {np.sqrt(test_mse_lgb):.2f}")
    print(f"Tuned LightGBM Test R² Score: {test_r2_lgb:.4f}")

    # Save LightGBM model
    joblib.dump(best_lgb, 'lightgbm_model.pkl')
    print("LightGBM model saved as 'lightgbm_model.pkl'")
except Exception as e:
    print(f"Error in LightGBM RandomizedSearchCV: {e}")
    print("Debug: Expected Features:", preprocessing.get_feature_names_out().shape[0])
    print("Debug: Housing Train Columns:", housing_train.shape[1])

# Visualize Predictions and Errors for All Models
models = {
    'Linear Regression': (test_predictions, 'test_predictions_error_lr.png') if 'test_predictions' in locals() else None,
    'Random Forest': (best_forest_test_pred, 'test_predictions_error_rf.png') if 'best_forest_test_pred' in locals() else None,
    'XGBoost': (best_xgb_test_pred, 'test_predictions_error_xgb.png') if 'best_xgb_test_pred' in locals() else None,
    'CatBoost': (best_cat_test_pred, 'test_predictions_error_cat.png') if 'best_cat_test_pred' in locals() else None,
    'LightGBM': (best_lgb_test_pred, 'test_predictions_error_lgb.png') if 'best_lgb_test_pred' in locals() else None
}

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
plt.style.use('seaborn-v0_8')  # Use a built-in seaborn style compatible with matplotlib

# Multi-Model Comparison Grid
fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 5)

for idx, (model_name, (predictions, _)) in enumerate(models.items()):
    if predictions is not None:
        percent_error = 100 * (predictions - labels_test) / labels_test
        residuals = labels_test - predictions

        # Scatter Plot
        ax_scatter = fig.add_subplot(gs[0, idx])
        sns.scatterplot(x=labels_test, y=predictions, alpha=0.3, color=colors[idx], ax=ax_scatter)
        ax_scatter.plot([labels_test.min(), labels_test.max()], [labels_test.min(), labels_test.max()], 'r--', lw=2)
        slope, intercept, r_value, _, _ = linregress(labels_test, predictions)
        ax_scatter.plot(labels_test, intercept + slope * labels_test, 'b-', label=f'Regression (R² = {r_value**2:.2f})')
        ax_scatter.set_xlabel("True Values", fontsize=12, fontweight='bold')
        ax_scatter.set_ylabel("Predicted Values", fontsize=12, fontweight='bold')
        ax_scatter.set_title(f"{model_name} Predictions", fontsize=14, fontweight='bold')
        ax_scatter.legend(fontsize=10)
        ax_scatter.grid(True, linestyle='--', alpha=0.7)

        # Boxplot for Error Distribution
        ax_box = fig.add_subplot(gs[1, idx])
        sns.boxplot(y=percent_error, color=colors[idx], ax=ax_box)
        ax_box.set_xlabel("Percent Error (%)", fontsize=12, fontweight='bold')
        ax_box.set_title(f"{model_name} Error Distribution", fontsize=14, fontweight='bold')
        ax_box.grid(True, linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()
plt.savefig("model_comparison_grid.png", dpi=300)
plt.close()

# Residual Plots
for idx, (model_name, (predictions, _)) in enumerate(models.items()):
    if predictions is not None:
        residuals = labels_test - predictions
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=predictions, y=residuals, alpha=0.3, color=colors[idx])
        plt.axhline(0, color='r', linestyle='--')
        plt.xlabel("Predicted Values", fontsize=12, fontweight='bold')
        plt.ylabel("Residuals", fontsize=12, fontweight='bold')
        plt.title(f"Residual Plot ({model_name})", fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f"residuals_{model_name.lower()}.png", dpi=300)
        plt.close()

# Feature Importance Visualization
if 'feature_importance_rf' in locals():
    fi_rf = feature_importance_rf.sort_values(ascending=False).head(15)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=fi_rf.values, y=fi_rf.index, hue=fi_rf.index, palette="viridis", dodge=False, legend=False)
    plt.title("Random Forest Feature Importance (Top 15)", fontsize=14, fontweight='bold')
    plt.xlabel("Importance Score", fontsize=12, fontweight='bold')
    plt.ylabel("Feature", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("feature_importance_rf.png", dpi=300)
    plt.close()

if 'feature_importance_xgb' in locals():
    fi_xgb = feature_importance_xgb.sort_values(ascending=False).head(15)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=fi_xgb.values, y=fi_xgb.index, hue=fi_xgb.index, palette="viridis", dodge=False, legend=False)
    plt.title("XGBoost Feature Importance (Top 15)", fontsize=14, fontweight='bold')
    plt.xlabel("Importance Score", fontsize=12, fontweight='bold')
    plt.ylabel("Feature", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("feature_importance_xgb.png", dpi=300)
    plt.close()

# Model Performance Table
results = []
for model_name, (preds, _) in models.items():
    if preds is not None:
        mse = mean_squared_error(labels_test, preds)  # Default is squared=True
        rmse = np.sqrt(mse)  # Manually calculate RMSE
        r2 = r2_score(labels_test, preds)
        results.append([model_name, rmse, r2])

print(tabulate(results, headers=["Model", "Test RMSE", "Test R²"], tablefmt="github"))

print("Script completed. Visualizations saved as PNG files.")