import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("D:\precision-agriculture-using-machine-learning-main\Data\Crop_recommendation.csv")  # Ensure correct file path

# Extract features and labels
X = data.drop(columns=["label"])  # Features (N, P, K, temperature, humidity, ph, rainfall)
y = data["label"]  # Target (crop name)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
crop_recommendation_model = RandomForestClassifier(n_estimators=100, random_state=42)
crop_recommendation_model.fit(X_train, y_train)

# Save the trained model correctly
model_path = r"D:\precision-agriculture-using-machine-learning-main\models\Crop_Recommendation.pkl"
with open(model_path, "wb") as file:
    pickle.dump(crop_recommendation_model, file)

print("✅ Model trained and saved successfully!")
# Load the model
with open(model_path, "rb") as file:
    crop_recommendation_model = pickle.load(file)

# Verify model type
print(f"✅ Model Loaded: {type(crop_recommendation_model)}")  

# Example input (replace with actual values)
data = np.array([[90, 42, 43, 20.5, 80, 6.5, 200]])  # (N, P, K, temp, humidity, pH, rainfall)

# Make prediction
prediction = crop_recommendation_model.predict(data)
print(f"🌾 Recommended Crop: {prediction[0]}")
