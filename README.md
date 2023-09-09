# Mammographic Masses Prediction README

## Overview:
This script utilizes various machine learning models to predict the severity of mammographic masses based on features such as age, mass shape, margin, and density.

## Libraries Used:
- pandas
- numpy
- sklearn (for preprocessing, model selection, and various classifiers)
- tensorflow (for Keras neural network model)

## Dataset:
The dataset used in this script has the following columns:
- BI-RADS assessment
- Age
- Mass Shape
- Margin
- Density
- Severity

The data is loaded from a local path (`/Users/amath/Downloads/MLCourse-2/mammographic_masses.data.txt`). 

## Data Preprocessing:
1. Columns with unknown values (`?`) are treated as NaN.
2. Rows with any NaN values are removed from the dataset.
3. Features are scaled using `StandardScaler` from scikit-learn.

## Machine Learning Models Used:
1. Decision Tree Classifier
2. Random Forest Classifier
3. Support Vector Machine (with various kernels: linear, rbf, sigmoid, poly)
4. K-Nearest Neighbors (tested for k values ranging from 1 to 50)
5. Multinomial Naive Bayes
6. Logistic Regression
7. Neural Network (using Keras)

### Neural Network Architecture:
- Input layer with 64 neurons (corresponding to 4 features)
- Dropout layer with 50% dropout rate
- Hidden layer with 64 neurons and ReLU activation
- Dropout layer with 50% dropout rate
- Output layer with 1 neuron and sigmoid activation (binary classification)

## Results:
After training each model, the script prints the accuracy of the model using 10-fold cross-validation.

## How to Use:
1. Ensure you have all the necessary libraries installed.
2. Replace the dataset path with the correct path on your machine.
3. Run the script. After execution, you'll see the accuracy results for each model.

## Future Enhancements:
1. Hyperparameter tuning for improved model accuracy.
2. Exploration of additional preprocessing steps, like feature engineering.
3. Inclusion of visualizations to understand the significance of each feature.
4. Saving the best-performing model for future predictions.
