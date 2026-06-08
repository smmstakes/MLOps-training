import os

import joblib
import pandas as pd

from src.config import DATA_PATH
from src.config import MODEL_DIR
from src.data_preprocessing import load_data
from src.data_preprocessing import preprocess_data
from src.model import split_data
from src.model import train_all_models

def main():

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

    churn_data = load_data(DATA_PATH)
    X, y = preprocess_data(churn_data)

    X_train, X_test, y_train, y_test = split_data(X, y)

    assert type(y_train) != pd.DataFrame, "y_train should be a Series, not a DataFrame"
    assert type(y_test) != pd.DataFrame, "y_test should be a Series, not a DataFrame"

    results = train_all_models(X_train, X_test, y_train, y_test)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(results.get("best_model"), os.path.join(MODEL_DIR, "best_model.pkl"))

if __name__ == "__main__":

    main()
