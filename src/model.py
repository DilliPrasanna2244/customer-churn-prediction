import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)


def train_and_evaluate(X_train, X_test, y_train, y_test, save_path='outputs/plots/'):
    """
    Train 3 models, print their accuracy, and return the best one.
    """
    os.makedirs(save_path, exist_ok=True)

    models = {
        "Logistic Regression": LogisticRegression(),

        # n_estimators=100 means 100 decision trees
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),

        # Builds trees sequentially, each fixing previous errors
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    best_model = None
    best_accuracy = 0
    best_name = ""

    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)

        # Predict on test data
        y_pred = model.predict(X_test)

        # Calculate accuracy
        acc = accuracy_score(y_test, y_pred)

        print(f"\n{'='*40}")
        print(f"Model: {name}")
        print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

        # Track best model
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_name = name

    print(f"\n🏆 Best Model: {best_name} with accuracy {best_accuracy*100:.2f}%")

    # --- Confusion Matrix for best model ---
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Churn', 'Churn'])
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix - {best_name}')
    plt.savefig(f'{save_path}confusion_matrix.png')
    plt.close()
    print("✅ Saved: confusion_matrix.png")

    return best_model, best_name


def plot_feature_importance(model, feature_names, save_path='outputs/plots/'):
    """
    Show which features matter most for prediction.
    Only works for tree-based models (Random Forest, Gradient Boosting).
    """
    if not hasattr(model, 'feature_importances_'):
        print("⚠️ Feature importance not available for this model")
        return

    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances.sort_values(ascending=False).head(10).plot(
        kind='bar', color='steelblue', figsize=(10, 5)
    )
    plt.title('Top 10 Features That Predict Churn')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.savefig(f'{save_path}feature_importance.png')
    plt.close()
    print("✅ Saved: feature_importance.png")


def save_model(model, scaler, filepath='outputs/models/'):
    """
    Save the trained model and scaler to disk using joblib.
    joblib is efficient for saving large numpy arrays inside models.
    """
    os.makedirs(filepath, exist_ok=True)
    joblib.dump(model, f'{filepath}churn_model.pkl')
    joblib.dump(scaler, f'{filepath}scaler.pkl')
    print(f"✅ Model and scaler saved to {filepath}")