import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load dataset
df = pd.read_csv("data.csv")

# Step 2: Drop unnecessary columns
df = df.drop(['UDI', 'Product ID'], axis=1)

# Step 3: Convert categorical to numeric
df = pd.get_dummies(df, columns=['Type'], drop_first=True)

# Step 4: Features and target
X = df.drop('Machine failure', axis=1)
y = df['Machine failure']

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42)

# Step 6: Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 7: Predict
y_pred = model.predict(X_test)

# Step 8: Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 9: Test with one sample
sample = X_test.iloc[0:1]
prediction = model.predict(sample)
print("Prediction (1=Failure, 0=Safe):", prediction[0])
import matplotlib.pyplot as plt


# Feature importance
importances = model.feature_importances_
features = X.columns
plt.barh(features, importances)
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.show()

import pickle
pickle.dump(model, open("model.pkl", "wb"))
model = pickle.load(open("model.pkl", "rb"))
prob = model.predict_proba(sample)
print("Failure Probability:", prob[0][1])