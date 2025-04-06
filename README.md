# Titanic Survival Prediction

This project uses the classic Titanic dataset to build a machine learning model that predicts whether a passenger survived or not.

## Project Overview

Based on the Titanic dataset from Kaggle, this project performs feature engineering and trains a logistic regression model to predict the survival of passengers in the test set. The final result is saved in a submission file named `titanic.csv`.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn

## Data Preprocessing

1. **Load Data**  
   Read `train.csv` and `test.csv` into Pandas DataFrames.

2. **Feature Selection and Encoding**  
   - Drop irrelevant features: `Name`, `Ticket`, and `Cabin`.
   - Apply one-hot encoding to categorical variables: `Sex` and `Embarked`.

3. **Missing Value Handling**  
   - Fill missing numeric values with the median.
   - Fill missing categorical values with the mode.
   - Fill other missing values with `False`.

4. **Feature Scaling**  
   - Standardize features using `StandardScaler`.

## Model Training

- A `LogisticRegression` model from Scikit-learn is used.
- The model is trained on the scaled training dataset (`X_train_scaled`, `y_train`) with `max_iter=1000`.

## Prediction and Output

- Predictions are made on the scaled test set (`X_test_scaled`).
- The output is saved in a CSV file `titanic.csv` with the following format:

