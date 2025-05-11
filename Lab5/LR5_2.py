import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

# Вхідні точки
x = np.array([0.1, 0.3, 0.4, 0.6, 0.7])
y = np.array([3.2, 3.0, 1.0, 1.8, 1.9])
# Побудова полінома 4-го ступеня
coeffs = np.polyfit(x, y, 4)
poly = np.poly1d(coeffs)
# Виведення коефіцієнтів
print("Коефіцієнти полінома (від вищого до нижчого ступеня):")
print(coeffs)
# Побудова графіка
x_vals = np.linspace(min(x), max(x), 200)
y_vals = poly(x_vals)
plt.plot(x_vals, y_vals, label='Інтерполяційний поліном', color='orange')
plt.scatter(x, y, color='green', label='Задані точки')
plt.title('Інтерполяція поліномом 4-го ступеня (Завдання 3)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
# Обчислення значень у проміжних точках
print(f"Значення полінома у x=0.2: {poly(0.2):.3f}")
print(f"Значення полінома у x=0.5: {poly(0.5):.3f}")
