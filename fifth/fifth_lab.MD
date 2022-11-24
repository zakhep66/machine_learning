```
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from tensorflow import keras
# from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow import keras


# from keras.utils import np_util

# Функция нормализации данных
def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x - min_val) / (max_val - min_val)
    return x


# стандартизация входных данных
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = normalize(X_train)
X_test = normalize(X_test)
X_train = X_train.reshape(-1, 32, 32, 3)

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

print(X_train.shape)

model = keras.Sequential([
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2), strides=2),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Flatten(),
    Dense(80, activation='relu'),
    Dense(10, activation='softmax')
])

# вывод структуры НС в консоль
print(model.summary())

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

his = model.fit(X_train, y_train, batch_size=32, epochs=5, validation_split=0.2)

model.evaluate(X_test, y_test)

```

Вывод

В ходе лабораторной работы была написана нейронная сеть, способная распознавать объекты на
изображении. Цели: научиться применять базовый функционал библиотеки Keras, научиться
моделировать свёрточные нейронные сети для распознавания объектов на изображении. Ознакомиться с основными методами
в работе со свёрточными нейронными сетями. Ознакомиться с популярными архитектурными решениями для задач
классификации изображений. Данные для распознавания были импортированы из библиотеки Cifar10. Смоделированная
свёрточная нейронная сеть распознаёт изображение с точностью (accuracy) примерно равной 70. Лабораторная работа
выполнена с помощью библиотек tensorflow и matplotlib.
