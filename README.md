# K-Nearest Neighbors with Kernel Smoothing

Этот проект реализует алгоритм классификации K - ближайших соседей (K-NN) с использованием сглаживания ядром(ядро Епаничникова) для оценки расстояний между объектами. В качестве примера используется набор данных из доступного датасета Iris.

## Описание:

Данный код выполняет классификацию с использованием метода k ближайших соседей. Для каждой точки в датасете рассчитывается количество ошибок классификации при различных значениях k с использованием функции ядра для сглаживания расстояний между точками.

## Требования:

Необходимо установить Python 3.12 с официального сайта (https://www.python.org/downloads/)
Для работы скрипта необходимо установить следующие библиотеки:

    numpy для работы с массивами.
    pandas для обработки данных и их сортировки.
    scikit-learn для загрузки набора данных Iris.

Вы можете установить необходимые библиотеки с помощью pip:

    pip install numpy pandas scikit-learn




### Запустите файл с помощью Python:

    python knn.py

Скрипт выведет массив, содержащий количество ошибок классификации для различных значений k.

## Подробности реализации:

Функция dist(a, b): Вычисляет евклидово расстояние между двумя точками a и b.
Функция ker(r): Применяет функцию ядра для сглаживания, используя значение расстояния r. В данной реализации используется эпанечиково ядро.

## Описание функций:

dist(a, b): Вычисляет евклидово расстояние между двумя точками a и b.
ker(r): Вычисляет значение ядерной функции для сглаживания на основе расстояния r.
Основной цикл перебирает различные значения k, вычисляет количество ошибок классификации и сохраняет результаты в массив Loo.


## Алгоритм:

Для каждого значения k из диапазона от 1 до 3*N+1 с шагом 3:
Вычисляется количество ошибок классификации.
Для каждой точки в датасете рассчитывается расстояние до всех остальных точек.
Сортируются точки по расстоянию, выбираются k ближайших соседей.
Для каждого класса рассчитывается вес на основе ядра, и выбирается класс с наибольшим весом.
Подсчитываются ошибки классификации и сохраняются в массив Loo.

## Результаты:

После выполнения кода будет напечатан массив Loo, который содержит количество ошибок классификации для различных значений k. Это поможет определить оптимальное значение k для данного набора данных.

##Примечания:
    
Значение N определяет количество проверяемых значений k. Подберите N в зависимости от размера вашего набора данных.
Код использует функцию эпанечикова ядра для сглаживания. Вы можете изменить ядро, если требуется другая функция.