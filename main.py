from src.data_preprocessing import load_data, clean_data, encode_features, split_and_scale
from src.eda import run_eda
from src.model import train_and_evaluate, plot_feature_importance, save_model

# ── Step 1: Load Data ──────────────────────────────────────
df = load_data('data/telco_churn.csv')

# ── Step 2: Clean Data ─────────────────────────────────────
df = clean_data(df)

# ── Step 3: Run EDA (saves charts to outputs/plots/) ───────
run_eda(df)

# ── Step 4: Encode Text Columns ────────────────────────────
df = encode_features(df)

# ── Step 5: Split & Scale ──────────────────────────────────
X_train, X_test, y_train, y_test, scaler, feature_names = split_and_scale(df)

# ── Step 6: Train Models & Evaluate ────────────────────────
best_model, best_name = train_and_evaluate(X_train, X_test, y_train, y_test)

# ── Step 7: Feature Importance ─────────────────────────────
plot_feature_importance(best_model, feature_names)

# ── Step 8: Save Model ─────────────────────────────────────
save_model(best_model, scaler)

print("\n🎉 Pipeline complete! Check outputs/ folder for all charts and saved model.")