# System-Threat-Forecaster

This repository contains the code and methodology for forecasting system threats using various machine learning models. The models implemented in this project include Support Vector Machine (SVM), K-Nearest Neighbors (KNN), and Gradient Boosting. The goal is to predict the likelihood of system threats based on historical data.

## Project Overview

- **Dataset**: The project uses a dataset containing system logs with various features such as `MachineID`, system metrics, and the target variable (`target`), which indicates whether a threat occurred.
- **Models Implemented**:
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Gradient Boosting Classifier
- **Objective**: Predict system threats using the historical dataset and evaluate the performance of different machine learning models.

## Dataset

The dataset consists of the following columns:
- `MachineID`: The unique identifier for each machine.
- `target`: The target variable representing whether a system threat occurred (binary classification).
- Various system metrics: Numeric features representing system performance metrics.

## Methodology

The approach for solving the classification problem follows a systematic sequence of steps, as outlined below:

### 1. **Data Preprocessing**

The first step involves data loading and cleaning. We begin by:
- **Loading the dataset**: Importing the dataset using pandas and inspecting its basic structure.
- **Handling Missing Data**: Identifying missing values and applying strategies like imputation or removal based on the data's nature.
- **Feature Selection**: Dropping irrelevant or redundant columns that do not contribute to model performance.
- **Feature Scaling**: Standardizing the numerical features using `StandardScaler` to ensure that all features are on the same scale, improving the performance of certain models like SVM and KNN.

### 2. **Model Training**

Once the data is preprocessed, we train multiple machine learning models to evaluate their performance:
- **Logistic Regression**: A simple linear model is trained to establish a baseline.
- **Support Vector Machine (SVM)**: A non-linear model is used to classify the data with a different approach, leveraging the kernel trick for higher-dimensional decision boundaries.
- **K-Nearest Neighbors (KNN)**: This model relies on the similarity between data points and is sensitive to feature scaling.
- **Gradient Boosting**: A powerful ensemble technique that combines weak learners (decision trees) to create a strong predictive model.

All models are trained using the training dataset, and their performance is evaluated using the validation dataset.

### 3. **Hyperparameter Tuning**

To improve the model's performance, we perform **hyperparameter tuning** for the Gradient Boosting model:
- A **Grid Search** with cross-validation is performed to find the best hyperparameters for the Gradient Boosting classifier.
- The hyperparameters tuned include `n_estimators`, `learning_rate`, `max_depth`, and `subsample`.
- The best hyperparameters are identified, and the model is re-trained with these optimal settings.

### 4. **Test Data Prediction**

Once the models are trained and evaluated, we use the best-performing model to make predictions on the **test data**:
- The test data undergoes preprocessing steps to ensure it aligns with the format used for training.
- Predictions are made on the test dataset using the trained model.
- The predicted values are then saved in a CSV file for submission.

### 5. **Model Evaluation**

Throughout the process, we evaluate model performance using **accuracy**. The accuracy score is computed for each model, and the best-performing model is selected based on this metric.

### 6. **Conclusion**

In the final section, we summarize the findings and discuss the results of the different models. The performance of each model is reviewed, and the next steps for potential improvements or further evaluation are outlined.


### Conclusion

This project implements three machine learning models for forecasting system threats. The models are evaluated based on accuracy, and the best-performing model can be selected for further analysis or deployment.

## License

This project is licensed under the MIT License.
