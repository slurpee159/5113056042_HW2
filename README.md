
# 5113056042_HW2

## Script Overview
The script follows the CRISP-DM framework for data mining and includes the following steps:

### 1. Data Loading and Preprocessing
Load the training and test datasets, clean them by removing unnecessary columns, and encode categorical features.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
import optuna
```

```python
# Load data
train_data = pd.read_csv(r"C:\Users\Rui\Desktop\5113056042_HW2\train.csv")
test_data = pd.read_csv(r"C:\Users\Rui\Desktop\5113056042_HW2\test.csv")
```

```python
# Drop unnecessary columns
train_data = train_data.drop(columns=["Name", "Ticket", "Cabin"])
test_data = test_data.drop(columns=["Name", "Ticket", "Cabin"])
```

```python
# Encode categorical columns
train_data["Sex"] = train_data["Sex"].map({"male": 0, "female": 1})
test_data["Sex"] = test_data["Sex"].map({"male": 0, "female": 1})
train_data = pd.get_dummies(train_data, columns=["Embarked"], drop_first=True)
test_data = pd.get_dummies(test_data, columns=["Embarked"], drop_first=True)
```

### 2. Handling Missing Values
Use median imputation for numeric columns and most frequent imputation for categorical columns.

```python
X = train_data.drop(columns=["Survived"])
y = train_data["Survived"]
```

```python
# Impute missing values
numeric_imputer = SimpleImputer(strategy="median")
categorical_imputer = SimpleImputer(strategy="most_frequent")
```

```python
# Impute numeric columns
X[X.select_dtypes(include=[np.number]).columns] = numeric_imputer.fit_transform(X.select_dtypes(include=[np.number]))
```

```python
# Impute categorical columns if any
non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns
if len(non_numeric_columns) > 0:
    X[non_numeric_columns] = categorical_imputer.fit_transform(X[non_numeric_columns])
```

```python
# Impute missing values in test data
test_numeric_columns = test_data.select_dtypes(include=[np.number]).columns
test_data[test_numeric_columns] = numeric_imputer.transform(test_data[test_numeric_columns])
test_non_numeric_columns = test_data.select_dtypes(exclude=[np.number]).columns
if len(test_non_numeric_columns) > 0:
    test_data[test_non_numeric_columns] = categorical_imputer.transform(test_data[test_non_numeric_columns])
```

### 3. Feature Selection
Use RFE and SelectKBest to select the best features for the model.

```python
def rfe_feature_selection(X, y, n_features=5):
    model = RandomForestClassifier(random_state=42)
    selector = RFE(model, n_features_to_select=n_features, step=1)
    selector = selector.fit(X, y)
    return selector.transform(X), selector

X_train_rfe, rfe_selector = rfe_feature_selection(X_train, y_train)

# SelectKBest feature selection
def select_k_best_features(X, y, k=5):
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    return selector.transform(X), selector

X_train_kbest, kbest_selector = select_k_best_features(X_train, y_train)
```

### 4. Hyperparameter Tuning with Optuna
Optimize hyperparameters for the Random Forest model using Optuna.

```python
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 200)
    max_depth = trial.suggest_int("max_depth", 2, 32)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    
    score = cross_val_score(model, X_train_kbest, y_train, cv=5, scoring="accuracy")
    return score.mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
```

### 5. Model Training and Evaluation
Train the Random Forest model with the best parameters from Optuna and evaluate it.

```python
best_params = study.best_params
model = RandomForestClassifier(**best_params, random_state=42)
model.fit(X_train_kbest, y_train)

# Evaluate on validation set
X_val_kbest = kbest_selector.transform(X_val)
y_pred = model.predict(X_val_kbest)
accuracy = accuracy_score(y_val, y_pred)
conf_matrix = confusion_matrix(y_val, y_pred)

print("Best Parameters from Optuna:", best_params)
print("Validation Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
```

### 6. Prediction and Submission
Make predictions on the test set and save them in the required Kaggle format.

```python
# Transform and predict on test set
X_test_kbest = kbest_selector.transform(test_data)
test_predictions = model.predict(X_test_kbest)

# Ensure PassengerId is int and create submission file
test_data["PassengerId"] = test_data["PassengerId"].astype(int)
submission = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": test_predictions})
submission.to_csv(r"C:\Users\Rui\Desktop\5113056042_HW2\submission.csv", index=False)
```

### Running the Script
Simply run the script as follows:

```bash
python 5113056042_HW2.py
```

```
kaggle competitions submit -c titanic -f submission.csv -m "My first submission"
```

![image](https://github.com/user-attachments/assets/0627f385-f5c8-4200-9a2c-04b0be37355e)



Make sure `train.csv` and `test.csv` are in the specified directory.
