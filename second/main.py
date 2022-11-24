# написать генератор кластеров их должно быть k штук
# допустим кластеров будет 5
# их нужно распределить по плоскости так, чтобы они не пересекались
import random

###################################################
# генератор
###################################################

import pandas
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

res = []

for i in range(10):
    while len(res) != 10:
        if (rand_cord := [random.randint(1, 4), random.randint(1, 4), 'red']) not in res:
            res.append(rand_cord)
for i in range(10):
    while len(res) != 20:
        if (rand_cord := [random.randint(5, 10), random.randint(1, 4), 'green']) not in res:
            res.append(rand_cord)
for i in range(10):
    while len(res) != 30:
        if (rand_cord := [random.randint(5, 10), random.randint(5, 10), 'blue']) not in res:
            res.append(rand_cord)
for i in range(10):
    while len(res) != 40:
        if (rand_cord := [random.randint(1, 4), random.randint(5, 10), 'black']) not in res:
            res.append(rand_cord)

some = pandas.DataFrame(res, columns=["x", "y", "color"])
some.to_csv('claster.csv', index=False)

###################################################
# расположение точек
###################################################
# нужно взять 8 точек и нарисовать на плоскости

import matplotlib.pyplot as plt
import pandas

claster = pandas.read_csv('claster.csv', delimiter=',')

x = claster['x'].tolist()  # x - координаты точек
y = claster['y'].tolist()  # y - координаты точек
color = claster['color'].tolist()

###################################################
# knn
###################################################
# оставшиеся 2 просто отобразить
# алгоритм должен посмотреть точки рядом и понять к какому кластеру их отнести
col_name = ['x', 'y', 'color']
dataset = pandas.read_csv('claster.csv')

x_train, x_test, y_train, y_test = train_test_split(dataset[col_name[:2]], dataset[col_name[2]], train_size=0.8,
                                                    random_state=1)

# модель класса, в которую передаем кол-во искомых соседей (k) и метрику
classifier = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
classifier.fit(x_train, y_train)  # а теперь обучаем модель

prediction = classifier.predict(x_test)  # предсказываем
print(prediction)  # выводим метки
print(y_test)

y_test = y_test.to_numpy()
counter_right_predictions = 0
for index in range(len(y_test)):
    if prediction[index] == y_test[index]:
        counter_right_predictions += 1

print("Процент правильных ответов:", counter_right_predictions / len(y_test) * 100)


fig1, ax1 = plt.subplots()
ax1.set_facecolor('white')  # цвет области Axes
ax1.set_title('knn + тестовые точки')  # заголовок для Axes


x_test = x_test.to_numpy()
counter = 0
for i in color:
    if i == 'red':
        ax1.scatter(x[counter], y[counter], c='red')  # цвет точек
    elif i == 'green':
        ax1.scatter(x[counter], y[counter], c='green')
    elif i == 'blue':
        ax1.scatter(x[counter], y[counter], c='blue')
    elif i == 'black':
        ax1.scatter(x[counter], y[counter], c='black')
    counter += 1


for i in range(8):
    ax1.scatter(*x_test[i], color='yellow')


fig1.set_figwidth(8)  # ширина и
fig1.set_figheight(8)  # высота "Figure"


plt.show()


# Вывод
# В этой лабораторной работе я познакомился с алгоритмом knn. Он нужен для определения k ближайших точек к
# заданной точке. Для реализации работы были использованы следующие библиотеки: pandas, matplotlib, sklearn Шанс с
# которым алгоритм предсказывает цвет точки примерно равен 92%.
