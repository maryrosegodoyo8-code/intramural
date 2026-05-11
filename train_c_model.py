import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
# Replace with your CSV file name

df = pd.read_csv('Csports.csv')

# Save encoders
label_encoders = {}

# Encode categorical columns
categorical_columns = [
    'match_id',
    'sport',
    'team_home',
    'team_away',
    'venue',
    'day_of_week',
    'time_slot',
    'weather_condition',
    'match_outcome'
]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
X = df.drop('match_outcome', axis=1)
y = df['match_outcome']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Save model and encoders
joblib.dump(model, 'sports_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

print('Model saved successfully!')