from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__, template_folder="C:/Users/admin/Desktop/browny-v1.0/templates")

# ✅ Load Model, Scaler, Encoder, and Feature Names
model_file = r"C:/Users/admin/Desktop/browny-v1.0/ml_model.pkl"

try:
    with open(model_file, "rb") as file:
        model, scaler, label_encoder, feature_names = pickle.load(file)  # ✅ Load feature names
    print("✅ Model, Scaler, and Encoder Loaded Successfully!")
except Exception as e:
    print(f"❌ Error Loading Model: {e}")

# ✅ Preprocess function (Ensure feature names match training)
def preprocess_input(data):
    """
    Convert form input into a format suitable for ML model.
    """
    df = pd.DataFrame([data])

    # ✅ Convert all feature names to lowercase to match training
    df.columns = df.columns.str.lower()

    # ✅ Ensure all missing columns are filled with 0 (prevents missing key errors)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0  # Add missing features

    # ✅ Reorder columns to match training
    df = df[feature_names]

    # ✅ Normalize data
    df = scaler.transform(df)

    return df

@app.route("/")
def index():
    return render_template("form.html")  # Flask looks inside `templates/`

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from frontend
        data = request.json

        # Preprocess input
        input_features = preprocess_input(data)

        # Make prediction
        prediction = model.predict(input_features)[0]

        # Convert numerical prediction back to label
        prediction_text = label_encoder.inverse_transform([prediction])[0]

        # Send response
        return jsonify({"prediction": prediction_text})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
