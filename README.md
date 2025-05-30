# AI/ML Internship - Task 3: Linear Regression - Housing Price Prediction

## Objective
The main objective of this task was to implement and understand simple and multiple linear regression using the Housing dataset to predict house prices.

## Dataset
The dataset used for this task is the [Housing.csv](Housing.csv) dataset, which contains various features related to house properties and their prices.

## Tools and Libraries Used
* **Python**
* **Pandas:** For data loading, preprocessing, and manipulation.
* **NumPy:** For numerical operations.
* **Scikit-learn:** For machine learning model implementation (Linear Regression, train-test split) and evaluation metrics.
* **Matplotlib:** For creating static visualizations.
* **Seaborn:** For making enhanced statistical graphics.

## Linear Regression Steps Performed:

### 1. Import and Preprocess the Dataset
* Loaded the `Housing.csv` dataset.
* Checked for missing values (none found).
* Identified and applied one-hot encoding to categorical features (e.g., `mainroad`, `furnishingstatus`), converting them into numerical format suitable for linear regression.
* **Key Insight:** Prepared the dataset with all numerical features for model training.

### 2. Split Data into Train-Test Sets
* Separated the target variable ('price') from the features (all other columns).
* Split the dataset into training (70%) and testing (30%) sets using `train_test_split`.
* **Key Insight:** Ensured a portion of the data remained unseen by the model for unbiased evaluation.

### 3. Fit a Linear Regression Model
* Initialized and trained a `LinearRegression` model from `sklearn.linear_model` on the training data (`X_train`, `y_train`).
* Extracted and displayed the learned coefficients for each feature and the model's intercept.
* **Key Insight:** The model learned the linear relationships between the features and the house price.

### 4. Evaluate Model Performance
* Made predictions on the unseen test set (`X_test`).
* Calculated common regression evaluation metrics:
    * **Mean Absolute Error (MAE):** Average absolute difference between actual and predicted values.
    * **Mean Squared Error (MSE):** Average of squared differences.
    * **Root Mean Squared Error (RMSE):** Square root of MSE, in the same units as the target.
    * **R-squared (R²):** Proportion of variance in the target explained by the model.
* **Key Insight:** Quantified the model's accuracy and explained variance on unseen data.

### 5. Plot Regression Line and Interpret Coefficients
* Generated a scatter plot of actual prices vs. predicted prices to visually assess model fit, where points closer to the diagonal line indicate better predictions.
* Reiterated the interpretation of feature coefficients, highlighting their influence on the predicted house price.
* **Key Insight:** Visual confirmation of prediction accuracy and understanding the direction and magnitude of each feature's impact on house price.

## Visualizations Included:
* `actual_vs_predicted_prices.png`

## What Was Learned:
Through this task, I gained practical experience in regression modeling, understanding and applying key evaluation metrics, and interpreting the coefficients of a linear regression model to explain feature importance. This forms a foundational understanding for predictive analytics.
