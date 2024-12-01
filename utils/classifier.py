# predict.py
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, make_scorer, f1_score
from sklearn.model_selection import KFold, RandomizedSearchCV
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
    test_predictions_mapped = ["OK" if pred == 1 else "NOK" for pred in test_predictions]
    # Load the original test data to add the predictions
    test_data = pd.read_csv(test_path)
    # Create a DataFrame with part_type_id and predictions
    result = pd.DataFrame({
        "part_type_id": test_data["part_type_id"],
        "prediction": test_predictions_mapped
    })
    # Save the predictions to a CSV file
    result.to_csv("test_predictions.csv", index=False, header=False)
    print("Predictions complete. Results saved to 'test_predictions.csv'.")


def find_best_model(X_train, y_train, n_iter=10, cv=5):
    """
    Find the best Random Forest model using RandomizedSearchCV.

    Parameters:
    - X_train: Preprocessed training data (features).
    - y_train: Training labels.
    - n_iter: Number of parameter settings sampled (default: 20).
    - cv: Number of cross-validation folds (default: 5).

    Returns:
    - best_rf: Best Random Forest model with tuned hyperparameters.
    - best_params: Best hyperparameters found during tuning.
    - duration: Time taken to run the search.
    """
    # Define the parameter grid
    param_dist = {
        'n_estimators': [50, 100, 200, 300],  # Number of trees in the forest
        'max_depth': [None, 10, 20, 30],  # Depth of the trees
        'min_samples_split': [2, 5, 10],  # Min samples required to split a node
        'min_samples_leaf': [1, 2, 4],  # Min samples required at each leaf node
        'max_features': ['sqrt', 'log2', None],  # Features to consider when looking for the best split
        'class_weight': ['balanced', 'balanced_subsample']  # Handles imbalanced data
    }
    # Random Forest classifier
    model = RandomForestClassifier(random_state=42)
    # Define the scoring metric (e.g., F1-score for imbalanced datasets)
    scorer = make_scorer(f1_score, average='binary')
    # RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scorer,
        cv=cv,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )
    # Fit RandomizedSearchCV
    random_search.fit(X_train, y_train)
    # Best parameters and model
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    print(f"Best parameters found: {best_params}")    
    return best_model, best_params
