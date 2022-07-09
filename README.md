# Machine_Learning_Templates
Python templates for different machine learning models. Features Regression, Classification, Clustering, Association Rules Learning, Reinforcement Learning, Natural Language Processing, Deep Learning, and Dimensionality Reduction models.

## Table of Contents:
- [Regression](#regression)

## Regression
Regression Modules Available:
- Decision Tree Regression
- Multiple Linear Regression
- Polynomial Regression
- Random Forest Regression
- Support Vector Regression

How to Access Regression:
- Call function model_wizard(path, size_select)
- path: path to csv file (string)
- size_select: size of training set (decimal)
- Specify Regression -> Specify Model

Notes:
- Polynomial Regression will prompt user for number of polynomial degrees to use in model
- Random Forest Regression will prompt user for number of estimators to use in model
- size_select will return r2 value as well as the regressor object, therefore regressor properties are available by setting a variable equal to the function (i.e. regressor = model_wizard(path, size_select))

[Table of Contents](#table-of-contents)
