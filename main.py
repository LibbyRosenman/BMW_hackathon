import pandas as pd
from utils.clean_data import preprocess_data
from utils.classifier import evaluate_model_kfold, generate_predictions_and_save, find_best_model
from utils.causal_analyzer import comprehensive_causal_inferance, create_dag, run_analysis, visualize_results, find_best_method


if __name__ == "__main__":
    results = []
    optional_preprocessing_methods = [(1,1), (1,2), (2,1), (2,2), (3,1)]
    scaling_dict = {1: "standard", 2: "min-max", "standard": 1, "min-max": 2}
    feature_selection_dict = {1: "baseline", 2: "changed_features", 3: "PCA", "baseline": 1, "changed_features": 2, "PCA": 3}
    for method in optional_preprocessing_methods:
        scaler = scaling_dict[method[1]]
        feature_selection_method = feature_selection_dict[method[0]]
        # Step 1: Preprocess data according to chosen method
        input_csv = "train.csv"
        output_csv = f"preprocess_train_{feature_selection_method}_{scaler}.csv"
        cleaned_df, pca_model, original_features = preprocess_data(input_csv, output_csv, filter_method=method[0], scaling_method=method[1], pca_components=None, pca_variance_threshold=0.95)
        # Step 2: Classification
        # Split into X (features) and y (target)
        X_train = cleaned_df.drop(columns=["status", "physical_part_id"])
        y_train = cleaned_df["status"]
        # train and evaluate the model on the data with k folds
        model, precision, recall, F1 = evaluate_model_kfold(X_train, y_train, k=5)
        results.append({
            "preprocessing_method": (feature_selection_dict[method[0]], scaling_dict[method[1]]),
            "precision": precision,
            "recall": recall,
            "F-1 score": F1
        })
    # Step 3: compare the methods and choose the best one.
    visualize_results(results)
    best_method_results = find_best_method(results)
    best_feature_selection = best_method_results["preprocessing_method"][0] # textual representation
    best_scaler = best_method_results["preprocessing_method"][1] # textual representation
    # Step 4: find best hyper parameters for the model
    final_cleaned_data = pd.read_csv(f"preprocess_train_{best_feature_selection}_{best_scaler}.csv")
    X_train_final = final_cleaned_data.drop(columns=["status", "physical_part_id"])
    y_train_final = final_cleaned_data["status"]
    best_model, best_params = find_best_model(X_train_final, y_train_final)
    # Save predictions for test set
    test_data = pd.read_csv("test.csv")
    test_feature_selection = feature_selection_dict[best_feature_selection]
    test_scaler = scaling_dict[best_scaler]
    test_data_cleaned = preprocess_data(test_data, "test_data_preprocess.csv", test_feature_selection, test_scaler)
    generate_predictions_and_save(best_model, test_data_cleaned, "test.csv")
    # Step 5: retrieve important features:
    is_pca = best_feature_selection == "PCA"
    important_features = run_analysis(best_model, X_train_final, is_pca, pca_model, original_features)
    # Step 6: causal inferance process:
    causes = list(important_features.keys())
    confounders = {}
    data = pd.read_csv(f"preprocess_train_{best_feature_selection}_{best_scaler}.csv")
    dag = create_dag(data, causes, confounders)
    causal_insights = comprehensive_causal_inferance(data, dag, causes)
    print(causal_insights)
    