import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Вхідний файл
input_file = 'income_data.txt'

# Читання даних
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = line.strip().split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data[:-1])
            y.append('<=50K')
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data[:-1])
            y.append('>50K')
            count_class2 += 1

X = np.array(X)
y = np.array(y)

# Кодування ознак
label_encoders = []
X_encoded = np.empty(X.shape, dtype=int)
for i in range(X.shape[1]):
    encoder = preprocessing.LabelEncoder()
    X_encoded[:, i] = encoder.fit_transform(X[:, i])
    label_encoders.append(encoder)

# Кодування класів
label_encoder_y = preprocessing.LabelEncoder()
y_encoded = label_encoder_y.fit_transform(y)

# Розбиття на train/test
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=5)

# Класифікатор
classifier = OneVsOneClassifier(SVC(kernel='rbf'))
classifier.fit(X_train, y_train)

# Оцінка моделі на тестовій вибірці
y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted')

# F1-міра на всій вибірці через cross_val_score
f1_cv = cross_val_score(classifier, X_encoded, y_encoded, scoring='f1_weighted', cv=3)

print("=== Оцінка класифікації ===")
print(f"Акуратність: {round(100 * accuracy, 2)}%")
print(f"Точність: {round(100 * precision, 2)}%")
print(f"Повнота: {round(100 * recall, 2)}%")
print(f"F1-міра: {round(100 * f1_cv.mean(), 2)}%")

# Передбачення для нової точки
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married',
              'Handlers-cleaners', 'Not-in-family', 'White', 'Male',
              '0', '0', '40', 'United-States']

input_data_encoded = []
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded.append(int(item))
    else:
        input_data_encoded.append(label_encoders[i].transform([item])[0])

input_data_encoded = np.array([input_data_encoded])
predicted_class = classifier.predict(input_data_encoded)
print("\nПередбачений клас для нового зразка:", label_encoder_y.inverse_transform(predicted_class)[0])
