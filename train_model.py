import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ✅ Load Dataset
dataset_path = r"C:/Users/admin/Desktop/browny-v1.0/migraine_data.csv"  
df = pd.read_csv(dataset_path)

# ✅ Convert all column names to lowercase
df.columns = df.columns.str.lower()

# ✅ Encode the Target Column (Convert migraine types to numerical labels)
le = LabelEncoder()
df["type"] = le.fit_transform(df["type"])  # Encoding categorical labels

# ✅ Split Features and Target
X = df.drop(columns=["type"])  # Features
y = df["type"]  # Target variable (migraine type)

# ✅ Split into Training & Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Normalize the Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ✅ Train the Model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Save the Model, Scaler, Label Encoder, and Feature Names
model_path = "C:/Users/admin/Desktop/browny-v1.0/ml_model.pkl"  
with open(model_path, "wb") as file:
    pickle.dump((model, scaler, le, list(X.columns)), file)  # Save feature names

print(f"✅ Model saved successfully at: {model_path}")
