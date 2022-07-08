# -*- coding: utf-8 -*-
"""
Creating a module to call on different machine learning models.
In creation stage... will likely split apart to different modules of their own.

Will also have a data cleaning module.

If available, data imaging will be available.

This will prompt the user to fill in information about the data, then it will
ask the user which type of analysis should be performed.

Information about the data requested:
    - Data ready for analysis (Y / N)
    - Columns of data
    - Columns of dependent data
    - Column(s) of independent data

Hints will be available about which type of predictive analytics platform to choose from.

The modules that will be called will have all of the options available... in lamens terms



There are nine main sections to call from:
    1. Regression
    2. Classification
    3. Clustering
    4. Association Rule Learning
    5. Reinforcement Learning
    6. Natural Language Processing
    7. Deep Learning
    8. Dimensionality Reduction
    9. Model Selection & Boosting

Each section will likely have their own components:
    1. Regression
        a. Decision Tree Regression
        b. Multiple Linear Regression
        c. Polynomial Regression
        d. Random Forest Regression
        e. Support Vector Regression
    2. Classification
        a. Decision Tree Classification
        b. K Nearest Neighbors Classification (KNN)
        c. Kernel SVM Classification
        d. Logistic Regression Classification
        e. Naive Bayes Classification
        f. Random Forest Classification
        g. Support Vector Machine Classification
    3. Clustering
        a. K Means Clustering
        b. Hierarchical Clustering
    4. Association Rule Learning
        a. Apriori
        b. Eclat
    5. Reinforcement Learning
        a. Upper Confidence Bound
        b. Thompson Sampling
    6. Natural Language Processing
        a. Natural Language Processing
    7. Deep Learning
        a. Artificial Neural Networks (ANN)
        b. Convolutional Neural Networks (CNN)
    8. Dimensionality Reduction
        a. (PCA)
        b. (LDA)
        C. (Kernel PCA)
    9. Model Selection & Boosting
        a. Model Selection - Grid Search
        b. Model Selection - K Fold Cross Validation
        c. Boosting - Xg Boost

- Klein
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def decision_tree_regression(path, size_select):
    dataset = pd.read_csv(path)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size_select, random_state = 0)

    # Training the Decision Tree Regression model on the Training set
    from sklearn.tree import DecisionTreeRegressor
    regressor = DecisionTreeRegressor(random_state = 0)
    regressor.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    np.set_printoptions(precision=2)
    print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

    # Evaluating the Model Performance
    from sklearn.metrics import r2_score
    score = r2_score(y_test, y_pred)
    print(f'The r2 score is {score}.')
    
    return regressor


def multiple_linear_regression(path, size_select):
    dataset = pd.read_csv(path)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size_select, random_state = 0)

    # Training the Multiple Linear Regression model on the Training set
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    np.set_printoptions(precision=2)
    print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
    
    # Evaluating the Model Performance
    from sklearn.metrics import r2_score
    score = r2_score(y_test, y_pred)
    print(f'The r2 score is {score}.')
    
    return regressor


def polynomial_regression(path, size_select):
    dataset = pd.read_csv(path)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    
    degree_select = int(input('Type polynomial degree to use\n'))

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size_select, random_state = 0)

    # Training the Polynomial Regression model on the Training set
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    poly_reg = PolynomialFeatures(degree = degree_select)
    X_poly = poly_reg.fit_transform(X_train)
    regressor = LinearRegression()
    regressor.fit(X_poly, y_train)
    
    # Predicting the Test set results
    y_pred = regressor.predict(poly_reg.transform(X_test))
    np.set_printoptions(precision=2)
    print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
    
    # Evaluating the Model Performance
    from sklearn.metrics import r2_score
    score = r2_score(y_test, y_pred)
    print(f'The r2 score is {score}.')
    
    return regressor


def random_forest_regression(path, size_select):
    dataset = pd.read_csv(path)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    
    estimators_select = int(input('Type number of estimators to use\n'))

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size_select, random_state = 0)

    # Training the Random Forest Regression model on the whole dataset
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = estimators_select, random_state = 0)
    regressor.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    np.set_printoptions(precision=2)
    print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
    
    # Evaluating the Model Performance
    from sklearn.metrics import r2_score
    score = r2_score(y_test, y_pred)
    print(f'The r2 score is {score}.')
    
    return regressor


def support_vector_regression(path, size_select):
    dataset = pd.read_csv(path)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    y = y.reshape(len(y), 1)
    
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size_select, random_state = 0)

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    y_train = sc_y.fit_transform(y_train)
    
    # Training the SVR model on the Training set
    from sklearn.svm import SVR
    regressor = SVR(kernel = 'rbf')
    regressor.fit(X_train, y_train.ravel())
    
    # Predicting the Test set results
    y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)).reshape(-1,1))
    np.set_printoptions(precision=2)
    print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
    
    # Evaluating the Model Performance
    from sklearn.metrics import r2_score
    score = r2_score(y_test, y_pred)
    print(f'The r2 score is {score}.')
    
    return regressor
    
    
def regression_wizard(path, size_select):
    regression_select = 0
    while regression_select not in ['a', 'b', 'c', 'd', 'e', 'f']:
        regression_select = input('Which regression would you like to use? (Type Letter):\n a. Decisiion Tree Regression \n b. Multiple Linear Regression \n c. Polynomial Regression \n d. Random Forest Regression \n e. Support Vector Regression \n f. EXIT \n')
        
    if regression_select == 'a':
        print('Decision Tree Regression Selected')
        model = decision_tree_regression(path, size_select)
    elif regression_select == 'b':
        print('Multiple Linear Regression Selected')
        model = multiple_linear_regression(path, size_select)
    elif regression_select == 'c':
        print('Polynomial Regression Selected')
        model = polynomial_regression(path, size_select)
        # have them pick degrees in the function
    elif regression_select == 'd':
        print('Random Forest Regression Selected')
        model = random_forest_regression(path, size_select)
        # have them pick number of estimators in the function
    elif regression_select == 'e':
        print('Support Vector Regression Selected')
        model = support_vector_regression(path, size_select)
    else:
        print('Leaving Model Wizard')
        
    return model
        
        
def model_wizard(path, size_select):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    model_select = 0
    while model_select not in [1, 2, 3, 4]:
        model_select = int(input('Which type of model would you like to use? (Type Number):\n 1. Regression \n 2. Classification \n 3. Clustering \n 4. EXIT \n'))
    
    if model_select == 1:
        print('Regression Selected')
        model = regression_wizard(path, size_select)
    elif model_select == 2:
        print('Classification Selected')
        # model = classification_wizard()
    elif model_select == 3:
        print('Clustering Selected')
        # model = clustering_wizard()
    else:
        print('Leaving Model Wizard')
        
    return model