import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import f_oneway
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


"""
status = df['status']
NOK_data = df[df['status'] == 'NOK']
exclude_columns = ['physical_part_id', 'status']
n = 6
numeric_df = NOK_data.drop(columns=exclude_columns)
print(f"performing Kmeans with K = {n} and euclidean distance metric:")
euclid_clusters, _ = perform_clustering_NOK(numeric_df, n_clusters=n)
print(f"performing Kmeans with K = {n} and pearson correlation distance metric:")
pearson_clusters, _ = perform_clustering_NOK(numeric_df, n_clusters=n, distance_metric="correlation")

numeric_df = df.drop(columns=exclude_columns)
print(f"performing Kmeans with K = {n} and euclidean distance metric:")
euclid_clusters, _ = perform_clustering(numeric_df, status, n_clusters=n)
print(f"performing Kmeans with K = {n} and pearson correlation distance metric:")
pearson_clusters, _ = perform_clustering(numeric_df, status, n_clusters=n, distance_metric="correlation")

# Add clusters to the NOK data
NOK_data['cluster'] = euclid_clusters
# NOK_data['cluster'] = pearson_clusters
OK_data = df[df['status'] == 'OK']
NOK_data.drop(columns=exclude_columns)
OK_data.drop(columns=exclude_columns)
compare_ok_nok(OK_data, NOK_data, top_features=5)
"""  
    
def kmeans_with_custom_distance(df, n_clusters=11, distance_metric='euclidean'):
    """_summary_
    Apply K-means clustering with a custom distance metric.
    Args:
        df: pandas dataframe.
        n_clusters: Number of clusters' Defaults to 11.
        distance_metric: the metric chosen to calculate similarity. Defaults to 'euclidean'.
        Optional distance metrics:
        "manhattan": Sum of absolute differences (L1 norm).
        "euclidean": L2 distance.
        "minkowski": Generalized distance metric, parameterized by the p value (default is 2, equivalent to Euclidean distance).
        "correlation": 1 minus the Pearson correlation coefficient.
    Returns:
        clusters, Kmeans: the partition to k stable clusters, and the model of K-means.
    """
    # Fit K-means using precomputed distances
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(df)
    return clusters, kmeans

# Function to visualize clusters in 2D
def visualize_clusters(df, clusters, status, title="Clusters Visualization"):
    """
    Visualize the clusters using PCA for dimensionality reduction.
    """
    # Apply PCA to reduce to 2 dimensions for visualization
    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(df)
    
    # Convert clusters and status to NumPy arrays (if they aren't already)
    clusters = np.array(clusters)
    status = np.array(status)
    
    # Define markers for clusters and colors for status
    markers = ['o', 's', 'x', '#', '*', '^', 'A', 'V']
    colors = {'OK': 'green', 'NOK': 'red'}
    
    plt.figure(figsize=(10, 6))
    
    for cluster in np.unique(clusters):
        for stat in np.unique(status):
            mask = (clusters == cluster) & (status == stat)  # Mask for current cluster and status
            plt.scatter(
                reduced_data[mask, 0], 
                reduced_data[mask, 1], 
                label=f"Cluster {cluster}, Status {stat}",
                alpha=0.7,
                marker=markers[cluster % len(markers)],  # Cycle through markers
                color=colors.get(stat, 'gray')  # Default to gray if status is unexpected
            )
    
    plt.title(title)
    plt.xlabel("Dimension 1 (t-SNE)")
    plt.ylabel("Dimension 2 (t-SNE)")
    plt.legend()
    plt.grid()
    plt.show()


def visualize_clusters_NOK(df, clusters, title="Clusters Visualization"):
    """
    Visualize the clusters using t-SNE for dimensionality reduction.
    """
    # Apply t-SNE to reduce to 2 dimensions for visualization
    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(df)
    
    # Convert clusters to a NumPy array (if it's a Series)
    clusters = np.array(clusters)
    
    # Plot the clusters
    plt.figure(figsize=(10, 6))
    for cluster in np.unique(clusters):
        mask = clusters == cluster  # Boolean mask for the current cluster
        plt.scatter(
            reduced_data[mask, 0], 
            reduced_data[mask, 1], 
            label=f"Cluster {cluster}", 
            alpha=0.7
        )
    plt.title(title)
    plt.xlabel("Dimension 1 (t-SNE)")
    plt.ylabel("Dimension 2 (t-SNE)")
    plt.legend()
    plt.grid()
    plt.show()


# Function to evaluate clustering quality
def evaluate_clustering(df, clusters, metric='euclidean'):
    """
    Evaluate clustering quality using silhouette score.
    """
    score = silhouette_score(df, clusters, metric=metric)
    print(f"Silhouette Score ({metric}): {score}")
    return score

# Main function to perform clustering and visualization
def perform_clustering(df, status, n_clusters=2, distance_metric='euclidean'):
    """
    Perform clustering using the specified distance metric and visualize results.
    """
    print(f"Performing clustering with {distance_metric} distance...")
    
    clusters, kmeans = kmeans_with_custom_distance(df, n_clusters=n_clusters, distance_metric=distance_metric)
    
    # Evaluate clustering
    evaluate_clustering(df, clusters, metric=distance_metric)
    
    # Visualize clusters
    visualize_clusters(df.values, clusters, status, title=f"K-means Clustering ({distance_metric} distance)")
    
    return clusters, kmeans


# Main function to perform clustering and visualization of NOK data
def perform_clustering_NOK(df, n_clusters=2, distance_metric='euclidean'):
    """
    Perform clustering using the specified distance metric and visualize results.
    """
    print(f"Performing clustering with {distance_metric} distance...")
    
    clusters, kmeans = kmeans_with_custom_distance(df, n_clusters=n_clusters, distance_metric=distance_metric)
    
    # Evaluate clustering
    evaluate_clustering(df, clusters, metric=distance_metric)
    
    # Visualize clusters
    visualize_clusters_NOK(df.values, clusters, title=f"K-means Clustering ({distance_metric} distance)")
    
    return clusters, kmeans


def analyze_nok_clusters(nok_data, ok_data, top_features=5):
    """
    Analyze NOK clusters to identify key features and differences from OK data.
    
    Parameters:
    - nok_data (DataFrame): Subset of data where status == 'NOK', including a 'cluster' column.
    - ok_data (DataFrame): Subset of data where status == 'OK'.
    - top_features (int): Number of top features to display for statistical and visual analysis.
    
    Returns:
    - feature_importance (DataFrame): DataFrame of feature importance from RandomForest.
    """
    # Ensure nok_data contains 'cluster'
    if 'cluster' not in nok_data.columns:
        raise ValueError("nok_data must include a 'cluster' column.")

    print("=== Performing Statistical Tests ===")
    # Perform ANOVA for each feature
    significant_features = []
    p_values = []
    for feature in nok_data.columns.drop(['cluster', 'status', 'physical_part_id']):
        groups = [group[feature].values for _, group in nok_data.groupby('cluster')]
        flattened_groups = [group.flatten() for group in groups]
        f_stat, p_val = f_oneway(*flattened_groups)
        if p_val < 0.05:  # Significant at 95% confidence level
            significant_features.append(feature)
            p_values.append(p_val)

    # Sort significant features by p-value
    significant_features_sorted = sorted(zip(significant_features, p_values), key=lambda x: x[1])
    print(f"Top {top_features} significant features across clusters:")
    for feature, p_val in significant_features_sorted[:top_features]:
        print(f"{feature}: p-value = {p_val}")

    print("\n=== Visualizing Key Features ===")
    # Visualize top significant features
    for feature, _ in significant_features_sorted[:top_features]:
        plt.figure(figsize=(8, 5))
        # sns.boxplot(x='cluster', y=feature, data=nok_data)
        plt.title(f"Feature: {feature} across Clusters")
        plt.show()

    print("\n=== Feature Importance Using Random Forest ===")
    # Prepare data for RandomForest
    nok_features = nok_data.drop(columns=['cluster', 'status', 'physical_part_id'])
    rf = RandomForestClassifier(random_state=42)
    rf.fit(nok_features, nok_data['cluster'])

    # Extract feature importance
    feature_importance = pd.DataFrame({
        'feature': nok_features.columns,
        'importance': rf.feature_importances_
    }).sort_values(by='importance', ascending=False)

    print("Top features distinguishing NOK clusters:")
    print(feature_importance.head(top_features))

    return feature_importance


def compare_ok_nok(ok_data, nok_data, top_features=5):
    print("\n=== Comparing NOK Clusters to OK Data ===")
    # Compare NOK clusters to OK data
    for cluster in nok_data['cluster'].unique():
        print(f"\nCluster {cluster} vs OK Data:")
        cluster_means = nok_data[nok_data['cluster'] == cluster].mean()
        differences = cluster_means - ok_data.mean()
        print(differences.sort_values(ascending=False).head(top_features))