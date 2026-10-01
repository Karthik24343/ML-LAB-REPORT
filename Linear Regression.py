import pandas as pd

data = {
    'size': [600, 800, 1000, 1200, 1400,
             1600, 1800, 2000, 2200, 2400],
    'price': [25, 32, 38, 45, 52,
              58, 65, 72, 80, 87]
}
df = pd.DataFrame(data)
print(df.head(10))

import matplotlib.pyplot as plt
plt.scatter(df['size'] , df['price'])
plt.xlabel('House Size')
plt.ylabel('Price')
plt.title('size vs price')
plt.show()

from sklearn.model_selection import train_test_split
X = df[['size']]
y = df['price']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print("Training data")
print(X_train)
print("Testing data")
print(X_test)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)

print("Intercept:",model.intercept_)
print("Coeffiecient:",model.coef_[0])

y_pred = model.predict(X_test)
result = pd.DataFrame({
    'Actual Price' : y_test.values,
    'Predicted Price':y_pred
})
print(result)

new_houses = pd.DataFrame({
    'size' : [1500,1700,2500]
})
predictions = model.predict(new_houses)
for size,price in zip(new_houses['size'],predictions):
  print(size,"sq feet =",round(price,2),"lakhs")

plt.scatter(df['size'],df['price'],label = 'Actual data')
plt.plot(df['size'],model.predict(df[['size']]),
        color = 'red' , label='regression Line')
plt.xlabel('House Size')
plt.ylabel('House Price')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()