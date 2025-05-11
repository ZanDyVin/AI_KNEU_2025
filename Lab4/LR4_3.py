import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

# Шлях до файлу з даними
input_file = 'C:/Codes/python codes/AIUniver/AI_KNEU_2025/Lab4/data_multivar_regr.txt'
data = np.loadtxt(input_file, delimiter=',')
# Розділення на X (всі колонки крім останньої) та y (остання колонка)
X = data[:, :-1]
y = data[:, -1]
# Розділення на тренувальні та тестові дані (80/20)
num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]
#Лінійна регресія
lin_regressor = linear_model.LinearRegression()
lin_regressor.fit(X_train, y_train)
y_pred_linear = lin_regressor.predict(X_test)

# Поліноміальна регресія
polynomial = PolynomialFeatures(degree=10)
X_train_transformed = polynomial.fit_transform(X_train)
datapoint = [[7.75, 6.35, 5.56]]
poly_datapoint = polynomial.fit_transform(datapoint)
poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, y_train)
print("\nLinear regression:\n", lin_regressor.predict(datapoint))
print("\nPolynomial regression:\n", poly_linear_model.predict(poly_datapoint))