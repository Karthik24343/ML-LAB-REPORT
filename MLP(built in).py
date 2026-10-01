from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

X = data.data
y = data.target

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_test)
print(X_test)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_test)
print(X_test)

from sklearn.neural_network import MLPClassifier
model = MLPClassifier(
    hidden_layer_sizes = (10,),
    max_iter = 1000,
    random_state=42
)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print(y_pred)

from sklearn.metrics import accuracy_score
print("Accuracy:",accuracy_score(y_test,y_pred))

from sklearn.metrics import confusion_matrix,classification_report
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))
print("Classification Report:")
print(classification_report(y_test,y_pred))