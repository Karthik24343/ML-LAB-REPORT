# logistic Regression(Built in)
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target
print(X.shape)
print(y.shape)
print(data.target_names)

print("Input Shape",X.shape)
print("OutPut shape",y.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

y_prob = model.predict_proba(X_test)
y_pred = model.predict(X_test)
print("Probabilities:")
print(y_prob[:5])
print("Predicted Labels:")
print(y_pred[:5])

import pandas as pd
from sklearn.metrics import accuracy_score
result = pd.DataFrame({
    'Actual' : y_test,
    'Predicted' : y_pred
})
print(result.head(10))
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy:" , accuracy)

thresholds = [0.3,0.5,0.7]
prob = model.predict_proba(X_test)[:, 1]
for t in thresholds:
  pred = (prob >= t).astype(int)
  accuracy = accuracy_score(y_test,pred)
  print("Threshold:" , t)
  print("Accuracy:" , accuracy)