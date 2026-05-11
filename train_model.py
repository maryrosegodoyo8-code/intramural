import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================================
# LOAD DATASET
# =========================================
df = pd.read_csv("intramural_sports_data.csv")

# =========================================
# REMOVE EXTRA SPACES IN COLUMN NAMES
# =========================================
df.columns = df.columns.str.strip()

# =========================================
# SHOW DATASET INFO
# =========================================
print("\nDATASET COLUMNS:")
print(df.columns)

print("\nFIRST 5 ROWS:")
print(df.head())

# =========================================
# FEATURES (EXACT COLUMN NAMES)
# =========================================
feature_columns = [
    "home_team_rank",
    "away_team_rank",
    "home_prev_wins",
    "away_prev_wins",
    "home_avg_score",
    "away_avg_score",
    "home_attendance_rate",
    "away_attendance_rate",
    "home_fatigue_index",
    "away_fatigue_index",
    "schedule_conflict_count",
    "days_since_last_match_home",
    "days_since_last_match_away",
    "referee_experience_years",
    "audience_count"
]

# =========================================
# TARGET COLUMN
# =========================================
target_column = "performance_score"

# =========================================
# INPUTS AND TARGET
# =========================================
X = df[feature_columns]
y = df[target_column]

# =========================================
# TRAIN TEST SPLIT
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================
# CREATE MODEL
# =========================================
model = LinearRegression()

# =========================================
# TRAIN MODEL
# =========================================
model.fit(X_train, y_train)

# =========================================
# MAKE PREDICTIONS
# =========================================
y_pred = model.predict(X_test)

# =========================================
# EVALUATE MODEL
# =========================================
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")
print("MAE  :", round(mae, 2))
print("RMSE :", round(rmse, 2))
print("R²   :", round(r2, 2))

# =========================================
# SAVE TRAINED MODEL
# =========================================
joblib.dump(model, "model.pkl")

print("\nModel trained successfully!")
print("Saved as: model.pkl")