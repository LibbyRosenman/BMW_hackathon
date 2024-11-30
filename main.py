import pandas as pd
from utils.clean_data import preprocess_data
from utils.classifier import evaluate_model_kfold, generate_predictions_and_save
from utils.causal_analyzer import comprehensive_causal_inferance, create_dag, visualize_dag, run_analysis, visualize_results, find_best_model


if __name__ == "__main__":
    results = []
    important_features = []
    models = {}
    optional_preprocessing_methods = [(1,1), (1,2), (2,1), (2,2), (3,1)]
    scaling_dict = {1: "standard", 2: "min-max", "standard": 1, "min-max": 2}
    feature_selection_dict = {1: "baseline", 2: "changed_features", 3: "PCA", "baseline": 1, "changed_features": 2, "PCA": 3}
    for method in optional_preprocessing_methods:
        scaler = scaling_dict[method[1]]
        feature_selection_method = feature_selection_dict[method[0]]
        # Step 1: Preprocess data according to chosen method
        input_csv = "train.csv"
        output_csv = f"preprocess_train_{feature_selection_method}_{scaler}.csv"
        cleaned_df = preprocess_data(input_csv, output_csv, filter_method=method[0], scaling_method=method[1], pca_components=None, pca_variance_threshold=0.95)
        # Step 2: Classification
        # Split into X (features) and y (target)
        X_train = cleaned_df.drop(columns=["status", "physical_part_id"])
        y_train = cleaned_df["status"].map({"OK": 1, "NOK": 0})
        # train and evaluate the model on the data with k folds
        model, precision, recall, F1 = evaluate_model_kfold(X_train, y_train, k=5)
        results.append({
            "preprocessing_method": (feature_selection_dict[method[0]], scaling_dict[method[1]]),
            "precision": precision,
            "recall": recall,
            "F-1 score": F1
        })
        models[(feature_selection_dict[method[0]], scaling_dict[method[1]])] = model
        # Step 3: Analyze feature importance
        important_features[(feature_selection_dict[method[0]], scaling_dict[method[1]])] = run_analysis(model, X_train)
    # Step 4: compare the models and choose the best one.
    visualize_results(results)
    best_model_results = find_best_model(results)
    best_feature_selection = best_model_results["preprocessing_method"][0]
    best_scaler = best_model_results["preprocessing_method"][1]
    best_model = models[(best_feature_selection, best_scaler)]
    # Step 5: causal inferance process:
    causes = important_features[best_model_results["preprocessing_method"]]
    
    # Save predictions for test set
    test_data = pd.read_csv("test.csv")
    test_feature_selection = feature_selection_dict[best_feature_selection]
    test_scaler = scaling_dict[best_scaler]
    test_data_cleaned = preprocess_data(test_data, "test_data_preprocess", test_feature_selection, test_scaler)
    generate_predictions_and_save(best_model, test_data_cleaned, "test.csv")