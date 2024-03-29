import math
import os

from matplotlib import pyplot as plt
import numpy as np
import matplotlib
import tensorflow as tf
from numba import prange

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # бесполезный код
matplotlib.use('TkAgg')  # бесполезный код


def tanh(arg):  # использую тангенс в качестве активации
    # return 1 / (1 + math.exp(1) ** (-x))
    return tf.nn.tanh(arg)


class DenseNN(tf.Module):
    def __init__(self, outputs):
        super().__init__()
        self.outputs = outputs  # количество нейронов, которое должно быть в полносвязном слое
        self.fl_init = False  # ФЛАГ

    def __call__(self, x):  # функтор
        if not self.fl_init:
            #  инициализация
            self.w = tf.random.truncated_normal((x.shape[-1], self.outputs), stddev=0.1,
                                                name="w")  # нормальная случайная величина, stddev - среднеквадратичное отклонение
            self.b = tf.zeros([self.outputs], dtype=tf.float32, name="b")  # вектор смещения

            self.w1 = tf.random.truncated_normal((self.outputs, self.outputs), stddev=0.1, name="w1")
            self.b1 = tf.zeros([self.outputs], dtype=tf.float32, name="b1")

            self.w2 = tf.random.truncated_normal((self.outputs, self.outputs), stddev=0.1, name="w2")
            self.b2 = tf.zeros([self.outputs], dtype=tf.float32, name="b2")

            self.w3 = tf.random.truncated_normal((self.outputs, self.outputs), stddev=0.1, name="w3")
            self.b3 = tf.zeros([self.outputs], dtype=tf.float32, name="b3")

            self.w6 = tf.random.truncated_normal((self.outputs, 1), stddev=0.1, name="w6")
            self.b6 = tf.zeros([1], dtype=tf.float32, name="b6")
            #  тензоры в переменные
            self.w = tf.Variable(self.w)
            self.b = tf.Variable(self.b)

            self.w1 = tf.Variable(self.w1)
            self.b1 = tf.Variable(self.b1)

            self.w2 = tf.Variable(self.w2)
            self.b2 = tf.Variable(self.b2)

            self.w3 = tf.Variable(self.w3)
            self.b3 = tf.Variable(self.b3)

            self.w6 = tf.Variable(self.w6)
            self.b6 = tf.Variable(self.b6)

            self.fl_init = True

        y = tanh(tanh(tanh(
            tanh(x @ self.w + self.b) @ self.w1 + self.b1) @ self.w2 + self.b2) @ self.w3 + self.b3) @ self.w6 + self.b6
        return y


model = DenseNN(4)  # количество нейронов

# обучающие данные
x_train = tf.random.uniform(minval=0, maxval=4 * math.pi, shape=(1000, 1))  # 1000 наблюдений
y_train = [np.sin(a / 2) for a in x_train]

# функция потерь
loss = lambda x, y: tf.reduce_mean(tf.square(x - y))  # квадрат рассогласования между требуемым и получившимся выходом
# должно быть yt - yt(предсказанное)
# оптимизатор для градиентного спуска
opt = tf.optimizers.Adam(learning_rate=0.001)  # шаг обучения 0.001
# opt = tf.optimizers.SGD(momentum=0.5, nesterov=True, learning_rate=0.01)

# алгоритм обучения сети
# подбор w и b

EPOCHS = 10  # итерации/эпохи/поколения
for n in prange(EPOCHS):
    for x, y in zip(x_train, y_train):  # перебор обучающего множества
        x = tf.expand_dims(x, axis=0)  # преобразование в матрицу 1 на кол-во наблюдений (2)
        y = tf.constant(y, shape=(1, 1))  # преобразование в матрицу 1 на 1
        # подавать игрик каждый раз на y(x_{t+1})

        with tf.GradientTape() as tape:
            f_loss = loss(y, model(x))
            # print('y', y ,'req', math.exp(x/2), 'model', model(x), 'loss', f_loss, sep = '\n')

            grads = tape.gradient(f_loss, model.trainable_variables)  # градиенты по всем обучаемым параметрам модели
            opt.apply_gradients(zip(grads,
                                    model.trainable_variables))  # применяем градиенты для реализации 1 шага
            # градиентного спуска (применяем к параметрам)

    print(f'поколение {n + 1}: {f_loss}')  # выводим потери

# print("sin(x/2):", np.sin(x/2))
print(model(tf.constant([[1.0]])))  # матрица входных данных (по кол-ву входов)

# График для сравнения

x = np.linspace(0, 4 * math.pi, 300)  # отрезок от нуля до двух пи, 100 точек
# создать два массива: первый будет состоять из массивов входов, во втором верные ответы
plt.scatter(x, np.sin(x / 2), s=1)
for i in x:
    plt.scatter(i, model(tf.constant([[i]], dtype=tf.float32)), c='r', s=1)
plt.show()

# нужно обучить модель как в третьей лабе учить модель нужно на игрик закидывать данные в случайно
# последовательности(массивы с игриками от Т) на вход идёт по три игрика, мы должны выяснить какой будет следующий
# игрик, после сместиться на один игрик и повторить операцию
