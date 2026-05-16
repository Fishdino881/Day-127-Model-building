# Day-127-Model-building

###  Overview

Day 127 focuses on **building a Machine Learning model** using prepared and processed data.

Model building is the stage where algorithms learn patterns from data and make predictions or decisions.

---

##  What is Model Building?

Model building involves:

* Selecting an algorithm
* Training the model on data
* Learning patterns and relationships
* Making predictions on unseen data

---

##  Model Building Workflow

```text id="mb127_1"
Data Collection → Preprocessing → Training → Prediction → Evaluation
```

---

##  Steps in Model Building

### 1️ Import Libraries

```python id="mb127_2"
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
```

---

### 2️ Load Dataset

```python id="mb127_3"
data = pd.read_csv("data.csv")
```

---

### 3️ Split Data

```python id="mb127_4"
X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)
```

---

### 4️ Train Model

```python id="mb127_5"
model = LogisticRegression()

model.fit(X_train, y_train)
```

---

### 5️ Make Predictions

```python id="mb127_6"
predictions = model.predict(X_test)
```

---

##  Common Algorithms

* Linear Regression
* Logistic Regression
* Decision Tree
* Random Forest
* SVM
* Neural Networks

---

##  Best Practices

- Use clean and balanced data
- Avoid overfitting
- Use cross-validation
- Tune hyperparameters

---

##  Benefits

* Learns patterns from data
* Automates predictions
* Supports intelligent decision-making

---

##  Challenges

* Poor-quality data affects accuracy
* Model selection can be difficult
* Overfitting and underfitting issues

---

##  Key Takeaways

- Model building is the core of ML
- Requires proper data preprocessing
- Training and evaluation are essential steps

