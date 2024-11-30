# predict.py
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier


def evaluate_model_kfold(X_train, y_train, k=5):
    """_summary_
        Perform K-Fold Cross Validation (K-Fold CV) on the training data.
        For each fold, calculate precision, recall, and F-1 score.
    Args:
        X_train (_type_): training data.
        y_train (_type_): training labels.
        k (int, optional): number of folds for cross-validation. Defaults to 5.

    Returns:
        the model, and all evaluation results.
    """
    precision_sum = 0
    recall_sum = 0
    F1_sum = 0
    kf = KFold(n_splits=k, shuffle=True, random_state=42)  # Shuffle data before splitting
    for train_index, val_index in kf.split(X_train):
        X_train_subgroup, X_valid = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_subgroup, y_valid = y_train.iloc[train_index], y_train.iloc[val_index]
        model = RandomForestClassifier(random_state=42, n_estimators=100)
        model.fit(X_train_subgroup, y_train_subgroup)
        y_pred = model.predict(X_valid)
        precision, recall, F1, _ = precision_recall_fscore_support(y_valid, y_pred, average='binary')
        precision_sum += precision
        recall_sum += recall
        F1_sum += F1
    # Calculate averages for all metrics
    precision_avg = precision_sum / k
    recall_avg = recall_sum / k
    F1_avg = F1_sum / k
    return model, precision_avg, recall_avg, F1_avg


def generate_predictions_and_save(model, test_data_cleaned, test_path):
    """_Summary_
    Generate predictions for the test data and save the results to a CSV file.
    
    Parameters:
    model: Trained Random Forest model.
    test_data_scaled: ndarray, Scaled test data features.
    test_path: Path to the test dataset (CSV).
    """
    # Generate predictions on the test data
    test_predictions = model.predict(test_data_cleaned)
    # Load the original test data to add the predictions
    test_data = pd.read_csv(test_path)
    # Create a DataFrame with part_type_id and predictions
    result = pd.DataFrame({
        "part_type_id": test_data["part_type_id"],
        "prediction": test_predictions
    })
    # Save the predictions to a CSV file
    result.to_csv("test_predictions.csv", index=False)
    print("Predictions complete. Results saved to 'test_predictions.csv'.")
