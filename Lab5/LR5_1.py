import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Вхідні дані
x = np.array([2, 4, 6, 8, 10, 12]).reshape(-1, 1)
y = np.array([6.5, 4.4, 3.8, 3.5, 3.1, 3.0])
# Побудова моделі
model = LinearRegression()
model.fit(x, y)
# Отримання коефіцієнтів
intercept = model.intercept_
slope = model.coef_[0]
print(f"Рівняння прямої: y = {intercept:.3f} + {slope:.3f}·x")
# Побудова графіка
plt.scatter(x, y, color='blue', label='Експериментальні дані')
plt.plot(x, model.predict(x), color='red', label='Лінія найменших квадратів')
plt.title('Метод найменших квадратів')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
