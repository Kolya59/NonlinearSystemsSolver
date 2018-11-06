from __future__ import division

from numpy import *
from sympy import *

sym_x, sym_y = symbols('x y')
symbol_f1, symbol_f2 = symbols('f1 f2', cls=Function)
init_printing(use_unicode=False, wrap_line=False)


# Метод простой итерации
def simple_iteration_method(x0, y0, f1, f2, eps, out_file):
    out_file.write('\nМетод простой итерации\n')
    xi = x0
    yi = y0
    # Вычисление производных
    df1dx = diff(f1, sym_x)
    df1dy = diff(f1, sym_y)
    df2dx = diff(f2, sym_x)
    df2dy = diff(f2, sym_y)
    # Создание кортежей соответствия переменных и  их значений
    con_list = [(sym_x, xi), (sym_y, yi)]
    # Составление Якобиана
    # TODO: Исправить типы в Якобиане
    jacobean = Matrix([df1dx, df1dy],
                      [df2dx, df2dy])
    # Вывод Якобиана
    out_file.write('\nЯкобиан\n{0}'.format(np.array2string(jacobean)).replace('[', '').replace(']', ''))
    # Вычисление числа обусловленности Якобиана
    jacobean_norm = linalg.norm(jacobean)
    out_file.write('\nНорма Якобиана = {0}'.format(jacobean_norm))
    i = 0
    xi = x0
    yi = y0
    out_file.write('\nItr   |   x   |   y   | Норма невязки | F1 | F2 | Норма Якобиана\n')
    while residual_norm >= eps:
        # Шаг итерации
        ++i
        # Вычисление xi,yi
        f1i = solve(f1, xi)
        f2i = solve(f2, yi)
        # Вычисление нормы Якобиана
        jacobean = Matrix([df1dx.subs(con_list), df1dy.subs(con_list)],
                          [df2dx.subs(con_list), df2dy.subs(con_list)]).astype(float)
        jacobean_norm = linalg.norm(jacobean)
        # Вычисление значения функций в точке xi,yi
        F1 = f2i - f1i
        F2 = f1i - f2i
        # Вычисление нормы невязки
        residual_norm = max(abs(f1i, f2i))
        # Вывод результатов
        out_file.write(i + ' ' + f2i + ' ' + f1i + ' ' + residual_norm + ' ' + F1 + ' ' + F2 + ' ' + jacobean_norm)
        # Вычисление следующего приближения
        xi = f2i
        yi = f1i


# Метод Ньютона
# a,b - границы отрезка, x0, xn  - вычисляемые приближения
def newton_method(x0, y0, f1, f2, eps, out_file):
    out_file.write('Метод Ньютона\n')
    i = 0
    xi = x = x0
    yi = y = y0
    # Вычисление производных
    df1dx = diff(f1, sym_x)
    df1dy = diff(f1, sym_y)
    df2dx = diff(f2, sym_x)
    df2dy = diff(f2, sym_y)
    # Создание кортежей соответствия переменных и  их значений
    con_list_i = [(sym_x, xi), (sym_y, yi)]
    con_list = [(sym_x, x), (sym_y, y)]
    # Составление Якобиана
    # TODO: Исправить типы в Якобиане
    jacobean = Matrix([df1dx, df1dy],
                      [df2dx, df2dy])
    # Вывод Якобиана
    out_file.write('\nЯкобиан\n{0}'.format(np.array2string(jacobean)).replace('[', '').replace(']', ''))
    # Вычисление нормы Якобиана
    norm = linalg.norm(jacobean)
    # Вывод заголовка
    out_file.write('\nItr   |   x   |   y   | Норма невязки | F1 | F2\n')
    # Вычисление значений с необходимой точностью
    while norm >= eps:
        # Шаг итерации
        ++i
        # Подстановка предыдущего приближения
        xi = x
        yi = y
        # Вычисление Якобиана
        jacobean = Matrix([df1dx, df1dy],
                          [df2dx, df2dy]).linalg.inv()
        # Вычисление нового приближения
        x = xi - jacobean[0][0] * f1.sub(con_list_i) - jacobean[0][1] * f2.sub(con_list_i)
        y = yi - jacobean[1][0] * f1.sub(con_list_i) - jacobean[1][1] * f2.sub(con_list_i)
        # Вычисление нормы невязки
        residual_norm = max(abs((x - xi), (y - yi)))
        # Вывод результатов
        out_file.write(i + ' ' + x + ' ' + y + ' ' + residual_norm + ' ' + f1.sub(con_list) + ' ' + f2.sub(con_list) + '\n')


# Метод градиентного спуска
def gradient_descent_method(x0, y0, f1, f2, eps, out_file):
    out_file.write('\n Метод градиентного спуска\n')
    i = 0
    # TODO: Что такое k
    k = 0.5
    # ff(x,y) = f1(x,y) + f2(x,y)
    ff = f1 + f2
    # Создание кортежей соответствия переменных и  их значений
    con_list0 = [(sym_x, x0), (sym_y, y0)]
    con_list = [(sym_x, x0), (sym_y, y0)]
    # TODO:Вывод градиента
    out_file.write()
    # Вывод заголовка
    out_file.write('\nItr   |   x   |   y   |   Alpha   | Норма невязки | F1 | F2 | FF | k |\n')
    # Вычисление нормы
    residual_norm = max()
    while residual_norm >= eps:
        # Итерационный шаг
        ++i
        x0 = x
        y0 = y
        alpha = dihotomia(f1, f2, ff, -10000, 100000, x0, y0, eps)
        x = x0 - alpha * diff(ff, x).sub(con_list0)
        y = y0 - alpha * diff(ff, y).sub(con_list0)
        residual_norm = max(fabs(diff(ff, x).sub(con_list)), fabs(diff(ff, y).sub(con_list)))
        # Вывод результатов
        out_file.write(i + ' ' + x + ' ' + y + ' ' + alpha + ' ' + residual_norm + ' ' + f1.sub(con_list) + ' ' + f2.sub(con_list) + ' ' + ff.sub(con_list) + ' ' + k + '\n')


# Метод половинного деления
def dihotomia(f1, f2, ff, a0, b0, x, y, eps):
    # Величина на которую мы отклонимся от середины отрезка
    delta = 0.5 * eps
    # Отрезок локализации минимума
    ak = a0, bk = b0
    # Пока длина отрезка больше заданной точности
    while (bk - ak) >= eps:
        # Точка, отличающаяся от середины на дельту
        lk = (ak + bk - delta) / 2
        mk = (ak + bk + delta) / 2
        # Проверка в какую часть попадает точка минимума слева от разбиения или справа и выбор соответствующей точки
        if g(f1, f2, x, y, lk) <= g(f1, f2, x, y, mk):
            # Правая граница отрезка локализации равна mk
            bk = mk
        else:
            # Левая граница отрезка локализации равна mk
            ak = lk
    # Возврат точки минимума
    return (ak + bk) / 2


def g(f1, f2, x, y, alpha):
    return (x - alpha * diff(f1, x).sub()) ** 2 + (y - alpha * diff(f2, y).sub()) ** 2
