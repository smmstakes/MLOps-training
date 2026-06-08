import os

import joblib
import matplotlib.pyplot as plt
import mlflow.sklearn  # or mlflow.xgboost if needed
import seaborn as sns
from mlflow.sklearn import log_model
from pandas import Series
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.config import MODEL_DIR
from src.config import RANDOM_SEED


def split_data(X, y):

    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)


def evaluate_model(model, X_test, y_test, model_name):

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Print classification report
    print(f"\nEvaluation for {model_name}")
    print(classification_report(y_test, y_pred))

    if y_proba is not None:

        print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")

    plt.show()


def train_all_models(X_train, X_test, y_train: Series, y_test: Series):

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, n_jobs=-1),
        "RandomForest": RandomForestClassifier(n_estimators=100,random_state=RANDOM_SEED, n_jobs=-1),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "SVM": SVC(probability=True, random_state=RANDOM_SEED),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=RANDOM_SEED)
    }

    best_model = None
    best_model_name = ""
    best_f1_score = 0.0

    mlflow.set_experiment("Churn Prediction Models")

    for model_name, model in models.items():

        with mlflow.start_run(run_name=model_name):

            print(f"Training {model_name}...")

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

            accuracy = float(accuracy_score(y_test, y_pred))
            f1 = float(f1_score(y_test, y_pred))
            auc_score = float(roc_auc_score(y_test, y_prob))

            if hasattr(model, "get_params"):

                params = model.get_params()
                filtered_params = {
                    key: value
                    for key, value in params.items()
                    if key in ["C", "n_estimators", "max_depth", "learning_rate"]
                }
                mlflow.log_params(filtered_params)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", auc_score)

            log_model(model, artifact_path=model_name)

            print(f"{model_name} - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, AUC: {auc_score:.4f}")

            if f1 > best_f1_score:

                best_f1_score = f1
                best_model = model
                best_model_name = model_name

    results = {
        "best_model": best_model,
        "best_model_name": best_model_name,
        "best_f1_score": best_f1_score
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODEL_DIR, f"{best_model_name}_best_model.pkl"))

    return results


def plot_roc_curves(models, X_test, y_test):

    plt.figure(figsize=(10, 8))
    for name, model in models.items():

        if hasattr(model, "predict_proba"):

            y_proba = model.predict_proba(X_test)[:, 1]

        elif hasattr(model, "decision_function"):

            y_proba = model.decision_function(X_test)

        else:

            continue

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random")

    plt.title("ROC Curve Comparison of Models")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")

    plt.grid()

    plt.show()
