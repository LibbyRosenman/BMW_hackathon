# BMW Hackathon - Predicting Quality of Car Parts

This project was developed during the BMW Hackathon Challenge at Constructor University. The goal was to enhance the accuracy of predictions regarding the status of car parts after examination and provide actionable insights to optimize the production process.

---

## Project Overview

**Main Goals:**
1. Improve prediction accuracy of car part statuses.
2. Identify critical features influencing predictions.
3. Deliver meaningful insights for optimizing production and reducing errors.

**Key Highlights:**
- Used advanced preprocessing techniques to clean and enhance the dataset.
- Built a robust classification model using Random Forest with fine-tuned hyperparameters.
- Leveraged causal inference for deeper insights into cause-and-effect relationships.

---

## Features and Workflow

1. **Data Preprocessing:**
   - Handled missing values and removed redundant features (low variance / high correlation to other features).
   - Scaled features using both standardization and min-max scaling.
   - Engineered features such as time in shift, temperature, and humidity.
   - Reduced dimensionality using PCA for some preprocessing scenarios.
   - Compared all combination of preprocessing methods by weighted score of precision, recall & F-1.

2. **Classification:**
   - Implemented a Random Forest model to handle imbalanced and high-dimensional data.
   - Optimized hyperparameters using RandomizedSearchCV.

3. **Feature Importance Analysis:**
   - Identified critical sensors in specific stations.
   - Highlighted the role of external conditions (e.g., temperature, humidity) and human involvement (e.g., time in shift).

4. **Causal Inference:**
   - Built a DAG and applied causal models using the DoWhy library.
   - Found that stabilizing environmental factors (e.g., temperature and humidity) reduces errors.

5. **Key Insights:**
   - Predictors include specific sensors, stations, and environmental conditions.
   - Optimization opportunities lie in reducing redundancy, tracking external factors, and refining models continuously.
