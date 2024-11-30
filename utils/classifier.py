# predict.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    return specificity

def evaluate_model_kfold(X_train, y_train, k=100):
    """
    Perform K-Fold Cross Validation (K-Fold CV) on the training data.
    For each fold, calculate accuracy, precision, recall, and specificity.
    """
    accuracy_sum = 0
    precision_sum = 0
    recall_sum = 0
    specificity_sum = 0
    kf = KFold(n_splits=k, shuffle=True, random_state=42)  # Shuffle data before splitting

    for train_index, val_index in kf.split(X_train):
        X_train_subgroup, X_valid = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_subgroup, y_valid = y_train.iloc[train_index], y_train.iloc[val_index]
        
        model = RandomForestClassifier(random_state=42, n_estimators=100)
        model.fit(X_train_subgroup, y_train_subgroup)
        
        y_pred = model.predict(X_valid)

        accuracy_sum += accuracy_score(y_valid, y_pred)
        precision, recall, _, _ = precision_recall_fscore_support(y_valid, y_pred, average='binary')
        precision_sum += precision
        recall_sum += recall
        specificity_sum += calculate_specificity(y_valid, y_pred)

    # Calculate averages for all metrics
    accuracy_avg = accuracy_sum / k
    precision_avg = precision_sum / k
    recall_avg = recall_sum / k
    specificity_avg = specificity_sum / k
    
    return accuracy_avg, precision_avg, recall_avg, specificity_avg


def generate_predictions_and_save(model, test_data_scaled, test_path):
    """
    Generate predictions for the test data and save the results to a CSV file.
    
    Parameters:
    model (RandomForestClassifier): Trained Random Forest model.
    test_data_scaled (ndarray): Scaled test data features.
    test_path (str): Path to the test dataset (CSV).
    """
    # Generate predictions on the test data
    test_predictions = model.predict(test_data_scaled)
    
    # Load the original test data to add the predictions
    test_data = pd.read_csv(test_path)
    test_data["Predicted_Output"] = test_predictions
    
    # Save the predictions to a CSV file
    test_data.to_csv("test_predictions.csv", index=False)
    print("Predictions complete. Results saved to 'test_predictions.csv'.")


def analyze_feature_importance(model, X_train):
    """
    Analyze the feature importance and display the top 10 most important features.
    
    Parameters:
    model (RandomForestClassifier): Trained Random Forest model.
    X_train (DataFrame): Raw training features (without 'status').
    """
    feature_importance = pd.DataFrame({
        'Sensor': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("Top sensors by importance:")
    print(feature_importance.head(10))
