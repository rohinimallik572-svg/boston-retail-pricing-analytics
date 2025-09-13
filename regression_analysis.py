import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv('../data/retail_data.csv')
X = data[['feature1', 'feature2']]
y = data['price']
model = LinearRegression()
model.fit(X, y)
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
