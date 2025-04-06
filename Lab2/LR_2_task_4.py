import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Вхідний файл
input_file = 'income_data.txt'

# Завантаження та попередня обробка
X, y = [], []
count_class1 = 0
count_class2 = 0
max_datapoints = 5000

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

# Кодування
label_encoders = []
X_encoded = np.empty(X.shape, dtype=int)
for i in range(X.shape[1]):
    encoder = preprocessing.LabelEncoder()
    X_encoded[:, i] = encoder.fit_transform(X[:, i])
    label_encoders.append(encoder)

label_encoder_y = preprocessing.LabelEncoder()
y_encoded = label_encoder_y.fit_transform(y)

# Train/test розбиття
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=5)

# Алгоритми
models = [
    ('LR', LogisticRegression(solver='liblinear')),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC(kernel='linear'))
]

# Оцінювання
results = {}
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1-score': f1_score(y_test, y_pred, average='weighted')
    }

# Вивід результатів
print("=== Порівняння алгоритмів ===")
for name in results:
    print(f"\n{name}:")
    for metric in results[name]:
        print(f"{metric}: {round(100 * results[name][metric], 2)}%")

# Побудова графіків
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
for metric in metrics:
    plt.figure()
    values = [results[model][metric] for model in results]
    plt.bar(results.keys(), values, color='skyblue')
    plt.title(f'{metric} порівняння')
    plt.ylabel(metric)
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.show()

# Найкраща модель
best_model = max(results.items(), key=lambda x: x[1]['F1-score'])
print(f"\nНайкраща модель за F1-score: {best_model[0]} ({round(100 * best_model[1]['F1-score'], 2)}%)")
