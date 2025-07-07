from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
import numpy as np

dataset = load_iris()
X,y = dataset.data, dataset.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Define parameter grid 
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
}

# Initialize the grid search
grid_search = GridSearchCV(estimator=GradientBoostingClassifier(),
                            param_grid=param_grid,
                            scoring='accuracy',
                            cv=5,
                            n_jobs=-1) 


# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

#Get the best model
best_model_grid = grid_search.best_estimator_
# Make predictions on the test set
y_pred = best_model_grid.predict(X_test)
# Evaluate the model
accuracy_grid = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy_grid)


#Define parameter distribution for RandomizedSearchCV
param_dist = {
    'C': np.logspace(-3, 3, 10),
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'] 
}

# Initialize the randomized search
random_search = RandomizedSearchCV(estimator=SVC(),
                                    param_distributions=param_dist,
                                    n_iter=20,
                                    scoring='accuracy',
                                    cv=5,
                                    n_jobs=-1,
                                    random_state=42)

# Fit the randomized search to the training data
random_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best Parameters (Randomized Search):", random_search.best_params_)
print("Best Score (Randomized Search):", random_search.best_score_)

# Get the best model
best_model_random = random_search.best_estimator_
# Make predictions on the test set
y_pred_random = best_model_random.predict(X_test)
# Evaluate the model
accuracy_random = accuracy_score(y_test, y_pred_random)
print("Test Accuracy (Randomized Search):", accuracy_random)