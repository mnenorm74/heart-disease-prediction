import pandas as pd
import numpy as np
import sklearn
import matplotlib
import keras
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn import model_selection
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import regularizers

data = pd.read_csv('./heart.csv')
# вывод первых 5 записей
print(data.head())
# вывод статистики
data.describe()
# гистограммы для каждого параметра
data.hist(figsize=(20, 20), facecolor='red')
plt.show()
# перекрестный график
pd.crosstab(data.age, data.target).plot(kind="bar", figsize=(20, 6))
plt.title('Частота сердечных заболеваний и возраст')
plt.xlabel('Возраст')
plt.ylabel('Частота')
plt.show()
# тепловая карта корреляция
plt.figure(figsize=(10, 10))
sns.heatmap(data.corr(), annot=True, fmt='.1f')
plt.show()
# возраст и максимальный пульс
age_unique = sorted(data.age.unique())
age_thalach_values = data.groupby('age')['thalach'].count().values
mean_thalach = []
for i, age in enumerate(age_unique):
    mean_thalach.append(sum(data[data['age'] == age].thalach) / age_thalach_values[i])
plt.figure(figsize=(10, 5))
sns.pointplot(x=age_unique, y=mean_thalach, color='red', alpha=0.8)
plt.xlabel('Возраст', fontsize=15, color='blue')
plt.xticks(rotation=45)
plt.ylabel('Максимальный пульс', fontsize=15, color='blue')
plt.title('Возраст и максимальный пульс', fontsize=15, color='blue')
plt.grid()
plt.show()

# Разделение данных на данные для обучения и тестирования.
X = np.array(data.drop(['target'], 1))
y = np.array(data['target'])
# нормализация
mean = X.mean(axis=0)
X -= mean
std = X.std(axis=0)
X /= std

# создание X и Y датасета для обучения
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

# преобразование данных в категориальные метки
Y_train = to_categorical(y_train, num_classes=None)
Y_test = to_categorical(y_test, num_classes=None)


def create_model():
    # create model
    model = Sequential()
    model.add(Dense(16, input_dim=13, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001),
                    activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(8, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(2, activation='softmax'))

    # компилирование модели
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model = create_model()
print(model.summary())

# обучение модели на тренировочных данных
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=50, batch_size=10)

model.evaluate(X_train, Y_train)
model.evaluate(X_test, Y_test)

# Точность модели
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Точность модели')
plt.ylabel('Точность')
plt.xlabel('Эпохи')
plt.legend(['train', 'test'])
plt.show()

# Ошибка модели
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Ошибка модели')
plt.ylabel('Ошибка')
plt.xlabel('Эпохи')
plt.legend(['train', 'test'])
plt.show()

# проверка на нулевом элементе
x = np.expand_dims(X_train[0], axis=0)
result = model.predict(x)
print(np.argmax(result))
# проверка на первом элементе
x = np.expand_dims(X_train[1], axis=0)
result = model.predict(x)
print(np.argmax(result))
