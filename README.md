# 🌸 Automated Hyperparameter Tuning on Iris Dataset

This project demonstrates automated hyperparameter tuning using `GridSearchCV` and `RandomizedSearchCV` on the classic Iris dataset with two classifiers: **Gradient Boosting Classifier** and **Support Vector Classifier (SVC)**.

---

## 📊 Dataset

- **Name**: Iris Dataset
- **Source**: `sklearn.datasets.load_iris()`
- **Features**:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
- **Target Classes**:
  - Iris-setosa
  - Iris-versicolour
  - Iris-virginica

---

## 🔧 Models & Tuning Techniques

### 1. 🔍 Grid Search with `GradientBoostingClassifier`

- Exhaustive search over a specified parameter grid.
- **Parameters Tuned**:
  - `n_estimators`: [50, 100, 200]
  - `learning_rate`: [0.01, 0.1, 0.2]
  - `max_depth`: [3, 5, 7]
- **Cross-Validation**: 5-fold

### 2. 🎲 Randomized Search with `SVC`

- Random sampling of hyperparameter combinations.
- **Parameter Distributions**:
  - `C`: log-uniform between 1e-3 and 1e3
  - `kernel`: ['linear', 'rbf', 'poly']
  - `gamma`: ['scale', 'auto']
- **Iterations**: 20
- **Cross-Validation**: 5-fold

---

## ✅ Results

### 🔹 Gradient Boosting Classifier (GridSearchCV)
- **Best Parameters**: Printed from `grid_search.best_params_`
- **Best Cross-Validation Score**: Printed from `grid_search.best_score_`
- **Test Accuracy**: Evaluated using `accuracy_score`

### 🔸 Support Vector Classifier (RandomizedSearchCV)
- **Best Parameters**: Printed from `random_search.best_params_`
- **Best Cross-Validation Score**: Printed from `random_search.best_score_`
- **Test Accuracy**: Evaluated using `accuracy_score`

---

## 💻 Requirements

Install required libraries using pip:

```bash
pip install scikit-learn numpy
