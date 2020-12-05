import pandas as pd
import numpy as np
import sklearn
import matplotlib
import keras
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns

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

