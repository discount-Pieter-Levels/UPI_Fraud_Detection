import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from typing import Dict, Tuple

app = Flask(__name__)

# HTML Template for Flask app
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Fraud Detection System</title>
</head>
<body>
    <h1>Fraud Detection System</h1>
    <form id="form" method="POST" action="/predict">
        <label for="amount">Transaction Amount:</label>
        <input type="number" id="amount" name="Amount" required><br><br>
        <button type="submit">Analyze Transaction</button>
    </form>
    <div id="result"></div>
</body>
</html>
'''

class MLFraudDetector:
    def __init__(self):
        self.model = XGBClassifier(n_estimators=100, max_depth=6, random_state=42, use_label_encoder=False, eval_metric='logloss')
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.categorical_columns = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain']

    def merge_datasets(self, transaction_path: str, identity_path: str) -> pd.DataFrame:
        """Merge transaction and identity datasets"""
        transactions = pd.read_csv(transaction_path)
        identity = pd.read_csv(identity_path)
        return transactions.merge(identity, on='TransactionID', how='left')

    def prepare_dataset(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare dataset for training"""
        # Fill missing values
        df.fillna(-999, inplace=True)

        # Encode categorical columns
        for col in self.categorical_columns:
            if col in df.columns:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))

        # Select features and target
        features = ['TransactionAmt', 'ProductCD'] + [col for col in self.categorical_columns if col in df.columns]
        X = df[features].values
        y = df['isFraud'].values

        return X, y

    def train(self, data_path: Tuple[str, str]) -> float:
        """Train the model"""
        try:
            df = self.merge_datasets(*data_path)
            X, y = self.prepare_dataset(df)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train model
            self.model.fit(X_train_scaled, y_train)
            predictions = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, predictions)

            print(f"Training Accuracy: {accuracy:.4f}")

            # Save the model and scaler
            joblib.dump(self.model, 'model/xgb_model.pkl')
            joblib.dump(self.scaler, 'model/scaler.pkl')

            return accuracy
        except Exception as e:
            print(f"Training error: {str(e)}")
            raise

    def predict(self, transaction: dict) -> dict:
        """Predict fraud probability for a transaction"""
        try:
            features = [transaction.get('Amount', 0)]
            scaled_features = self.scaler.transform([features])
            prediction = self.model.predict(scaled_features)[0]
            return {"prediction": "Fraudulent" if prediction == 1 else "Not Fraudulent"}
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return {"error": str(e)}

# Initialize the detector
detector = MLFraudDetector()

def train_model():
    """Train the model"""
    transaction_path = 'data/train_transaction.csv'
    identity_path = 'data/train_identity.csv'
    accuracy = detector.train((transaction_path, identity_path))
    print(f"Model trained with accuracy: {accuracy:.4f}")

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        result = detector.predict(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # Train model before running the server
    train_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
