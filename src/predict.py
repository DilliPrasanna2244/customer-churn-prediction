import joblib
import pandas as pd


def load_model(model_path='outputs/models/churn_model.pkl',
               scaler_path='outputs/models/scaler.pkl'):
    """Load the saved model and scaler from disk."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_single_customer(customer_data: dict, model, scaler):
    """
    Predict churn for a single customer.
    customer_data: dictionary with feature values
    """
    df = pd.DataFrame([customer_data])
    df_scaled = scaler.transform(df)
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]

    result = "⚠️ WILL CHURN" if prediction == 1 else "✅ WILL STAY"
    print(f"\nPrediction: {result}")
    print(f"Churn Probability: {probability*100:.2f}%")
    return prediction, probability