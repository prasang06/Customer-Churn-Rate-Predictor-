# Customer Churn Prediction using PyTorch

## Overview

This project develops a machine learning pipeline to predict customer churn using demographic information, subscription details, customer engagement metrics, and spending behavior.

The workflow covers the complete machine learning lifecycle, including:

* Exploratory Data Analysis (EDA)
* Data preprocessing
* Feature engineering
* Baseline machine learning models
* Artificial Neural Networks (ANNs) using PyTorch
* Model evaluation and comparison

The primary objective is to classify whether a customer is likely to churn (leave the service) based on historical customer data.

---

## Dataset

The dataset contains customer-related information such as:

| Feature           | Description                                 |
| ----------------- | ------------------------------------------- |
| Age               | Customer age                                |
| Gender            | Customer gender                             |
| Tenure            | Length of customer relationship             |
| Usage Frequency   | Service usage frequency                     |
| Support Calls     | Number of support interactions              |
| Payment Delay     | Payment delay history                       |
| Subscription Type | Customer subscription tier                  |
| Contract Length   | Contract duration                           |
| Total Spend       | Total customer expenditure                  |
| Last Interaction  | Most recent customer interaction            |
| Churn             | Target variable (0 = Retained, 1 = Churned) |

CustomerID was removed during preprocessing because it serves only as an identifier and does not provide predictive business value.

---

## Exploratory Data Analysis

Several analyses were performed to understand the dataset:

* Class distribution analysis
* Feature distribution visualization
* Correlation analysis
* Investigation of relationships between features and churn

### Key Observations

* The dataset is approximately balanced between churned and retained customers.
* Payment Delay and Support Calls exhibit strong positive relationships with churn.
* Usage Frequency and Total Spend show negative relationships with churn.
* Many numerical features exhibit nearly uniform distributions.

These observations suggest that the dataset may be synthetically generated rather than collected from a real-world customer base.

---

## Data Preprocessing

The following preprocessing pipeline was applied:

1. Removed CustomerID
2. One-hot encoded categorical variables
3. Split data into training and testing sets
4. Standardized numerical features using StandardScaler
5. Converted data into PyTorch tensors

This preprocessing ensures compatibility with neural network training and prevents data leakage.

---

## Models Implemented

### Logistic Regression

A Logistic Regression model was trained as a baseline classifier.

This establishes a benchmark against which more complex models can be compared.

### Artificial Neural Network (PyTorch)

The neural network architecture consists of:

Input Layer

→ Dense (64)

→ ReLU

→ Dense (32)

→ ReLU

→ Dense (16)

→ ReLU

→ Dense (1)

→ Sigmoid

Training configuration:

* Loss Function: Binary Cross Entropy Loss (BCELoss)
* Optimizer: Adam
* Learning Rate: 0.001
* Framework: PyTorch

### XGBoost

An XGBoost classifier was also trained to compare performance against both the baseline model and the neural network.

---

## Results

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | ~83%     |
| PyTorch ANN         | ~91%     |
| XGBoost             | ~100%    |

### ANN Classification Performance

| Metric    | Score |
| --------- | ----- |
| Accuracy  | ~91%  |
| Precision | ~90%  |
| Recall    | ~91%  |
| F1 Score  | ~90%  |

The neural network significantly outperformed the Logistic Regression baseline, indicating the presence of nonlinear relationships within the data.

---

## Discussion

While the ANN demonstrated strong predictive performance, the near-perfect performance achieved by XGBoost is unusual for a real-world churn prediction task.

Possible explanations include:

* Synthetic dataset generation
* Highly separable feature distributions
* Strong deterministic relationships between features and the target variable

As a result, the reported metrics should be interpreted as a demonstration of machine learning techniques rather than an estimate of real-world production performance.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* PyTorch
* XGBoost
* Jupyter Notebook

---

## Future Improvements

Potential enhancements include:

* Hyperparameter optimization
* Cross-validation
* SHAP-based feature importance analysis
* Model deployment using Streamlit
* Testing on real-world churn datasets
* Experimentation with deeper neural network architectures

---

## Conclusion

This project demonstrates an end-to-end machine learning workflow for customer churn prediction using both classical machine learning and deep learning techniques.

The results show that neural networks can significantly improve predictive performance over linear models, while also highlighting the importance of critically evaluating dataset quality and model performance before drawing business conclusions.
