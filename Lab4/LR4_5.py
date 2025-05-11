import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
# Генерація даних
m = 100
X = 6 * np.random.rand(m, 1) - 5
y = 0.7 * X ** 2 + X + 3 + np.random.randn(m, 1)
# Розділення на тренувальні та тестові дані
num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]
# Лінійна регресія
lin_regressor = linear_model.LinearRegression()
lin_regressor.fit(X_train, y_train)
y_pred_linear = lin_regressor.predict(X_test)
# Поліноміальна регресія
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)
poly_regressor = linear_model.LinearRegression()
poly_regressor.fit(X_train_poly, y_train)
y_pred_poly = poly_regressor.predict(X_test_poly)
# Сортування для гладкої кривої
X_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
X_range_poly = poly_features.transform(X_range)
y_range_poly = poly_regressor.predict(X_range_poly)
# Побудова графіка
plt.scatter(X_test, y_test, color='green', label='Тестові дані')
plt.plot(X_test, y_pred_linear, color='black', linewidth=2, label='Лінійна регресія')
plt.plot(X_range, y_range_poly, color='blue', linewidth=2, label='Поліноміальна регресія')
plt.legend()
plt.xlabel("X")
plt.ylabel("y")
plt.title("Лінійна та Поліноміальна регресії")
plt.show()
