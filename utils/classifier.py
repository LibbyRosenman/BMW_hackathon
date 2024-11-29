# classifier.py
from predict import load_and_preprocess_data, train_and_evaluate_model, generate_predictions_and_save, analyze_feature_importance

def main(train_path, test_path):
    """
    Main function to load data, train model, and make predictions.
    
    Parameters:
    train_path (str): Path to the training data file (CSV).
    test_path (str): Path to the test data file (CSV).
    """
    # Step 1: Load and preprocess data
    X_train_scaled, y_train, test_data_scaled, X_train = load_and_preprocess_data(train_path, test_path)
    
    # Step 2: Train and evaluate the model
    model = train_and_evaluate_model(X_train_scaled, y_train)
    
    # Step 3: Generate predictions and save results
    generate_predictions_and_save(model, test_data_scaled, test_path)
    
    # Step 4: Analyze feature importance
    analyze_feature_importance(model, X_train)

# Example usage:
main("cleaned_X.csv", "test.csv")
