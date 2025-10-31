from flask import Flask, request, jsonify, render_template
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from collections import Counter

app = Flask(__name__)

DATA_PATH = "adhd_dataset_age9_24_flipped_risk.csv"
RF_MODEL_PATH = "rf_model.pkl"
SVM_MODEL_PATH = "svm_model.pkl"
LR_MODEL_PATH = "log_reg_model.pkl"

def load_or_train_models():
    if os.path.exists(RF_MODEL_PATH) and os.path.exists(SVM_MODEL_PATH) and os.path.exists(LR_MODEL_PATH):
        print("✅ Loading pre-trained models...")
        rf_model = joblib.load(RF_MODEL_PATH)
        svm_model = joblib.load(SVM_MODEL_PATH)
        log_reg_model = joblib.load(LR_MODEL_PATH)
    else:
        print("⚙️ Training models for the first time...")
        df = pd.read_csv(DATA_PATH)

        X = df[['Age', 'Attention Score', 'Hyperactivity Score', 'Impulsivity Score', 'Sleep Hours', 'Academic Score']]
        y = df['Predicted_Risk']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf_model = RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)

        svm_model = LinearSVC(random_state=42, max_iter=2000)
        svm_model.fit(X_train, y_train)

        log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
        log_reg_model.fit(X_train, y_train)

        joblib.dump(rf_model, RF_MODEL_PATH)
        joblib.dump(svm_model, SVM_MODEL_PATH)
        joblib.dump(log_reg_model, LR_MODEL_PATH)

    return rf_model, svm_model, log_reg_model

rf_model, svm_model, log_reg_model = load_or_train_models()

# ===============================
# Routes
# ===============================
@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/awareness')
def awareness():
    return render_template('index.html')  # can link to awareness page later


# ✅ NEW POST ROUTE — the actual prediction API
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json()

    input_df = pd.DataFrame([[ 
        data['age'],
        data['attention_score'],
        data['hyperactivity_score'],
        data['impulsivity_score'],
        data['sleep_hours'],
        data['academic_score']
    ]], columns=['Age', 'Attention Score', 'Hyperactivity Score', 'Impulsivity Score', 'Sleep Hours', 'Academic Score'])

    rf_pred = rf_model.predict(input_df)[0]
    svm_pred = svm_model.predict(input_df)[0]
    log_reg_pred = log_reg_model.predict(input_df)[0]

    preds = [rf_pred, svm_pred, log_reg_pred]
    final_pred = Counter(preds).most_common(1)[0][0]

    return jsonify({
        "Random Forest": rf_pred,
        "SVM": svm_pred,
        "Logistic Regression": log_reg_pred,
        "Ensemble Majority Vote": final_pred
    })


if __name__ == '__main__':
    app.run(debug=True)
