from __future__ import division

from numpy import *
from sympy import *

sym_x, sym_y = symbols('x y')
symbol_f1, symbol_f2 = symbols('f1 f2', cls=Function)
init_printing(use_unicode=False, wrap_line=False)


# Метод простой итерации
def simple_iteration_method(x0, y0, f1, f2, eps, out_file):
    out_file.write('\nМетод простой итерации\n')
    i = 0
    xi = x0
    yi = y0
    # Вычисление производных
    df1dx = diff(f1, sym_x)
    df1dy = diff(f1, sym_y)
    df2dx = diff(f2, sym_x)
    df2dy = diff(f2, sym_y)
    # Создание кортежей соответствия переменных и  их значений
    con_list_i = [(sym_x, xi), (sym_y, yi)]
    # Создание матрицы Якоби
    jacobi_matrix = Matrix([[df1dx, df1dy],
                            [df2dx, df2dy]])
    # Вывод Якобиана
    out_file.write('\nМатрица Якоби\n{0}'
                   .format(np.array2string(np.array(jacobi_matrix)))
                   .replace('[', '').replace(']', ''))
    # Подстановка x0 и y0 в матрицу Якоби
    value_of_jacobi_matrix = jacobi_matrix.subs(con_list_i)
    # Вывод значений матрицы Якоби
    out_file.write('\nЗначение матрицы Якоби в точке (x0,y0)\n{0}'
                   .format(np.array2string(np.array(value_of_jacobi_matrix).astype(float64)))
                   .replace('[', '').replace(']', ''))
    # Вычисление нормы матрицы Якоби
    jacobi_matrix_norm = linalg.norm(value_of_jacobi_matrix, np.inf)
    # Вывод нормы матрицы Якоби
    out_file.write('\nНорма матрицы Якоби = {0}'.format(jacobi_matrix_norm))
    # Вычисление нормы невязки
    residual_norm = max(abs(solve(f1.subs(sym_x, x0))[0]), abs(solve(f2.subs(sym_y, y0))[0]))
    # Вывод заголовка
    out_file.write(
        '\nItr |   x           |       y       | Норма невязки     |     F1        |       F2        | Норма Якобиана\n')
    while residual_norm >= eps:
        # Шаг итерации
        i += 1
        # Вычисление xi,yi
        y = solve(f1.subs(sym_x, xi))[0]
        x = solve(f2.subs(sym_y, yi))[0]
        # Создание кортежей соответствия переменных и  их значений
        con_list = [(sym_x, x), (sym_y, y)]
        con_list_i = [(sym_x, xi), (sym_y, yi)]
        # Вычисление нормы Якобиана
        jacobi_matrix_norm = linalg.norm(jacobi_matrix.subs(con_list_i), np.inf)
        # Вычисление значения функций в точке xi,yi
        f1_value = f1.subs(con_list)
        f2_value = f2.subs(con_list)
        # Вычисление нормы невязки
        residual_norm = max(abs(f1_value), abs(f2_value))
        # Вывод результатов
        out_file.write(
            '{0} {1} {2} {3} {4} {5} {6}\n'.format(i, x, y, residual_norm, f1_value, f2_value, jacobi_matrix_norm))
        # Вычисление следующего приближения
        xi = x
        yi = y


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
    # Создание матрицы Якоби
    jacobi_matrix = Matrix([[df1dx, df1dy],
                            [df2dx, df2dy]])
    # Вывод Якобиана
    out_file.write('\nМатрица Якоби\n{0}'
                   .format(np.array2string(np.array(jacobi_matrix)))
                   .replace('[', '').replace(']', ''))
    # Подстановка x0 и y0 в матрицу Якоби
    value_of_jacobi_matrix = jacobi_matrix.subs(con_list_i)
    # Вывод значений матрицы Якоби
    out_file.write('\nЗначение матрицы Якоби в точке (x0,y0)\n{0}'
                   .format(np.array2string(np.array(value_of_jacobi_matrix).astype(float64)))
                   .replace('[', '').replace(']', ''))
    # Вычисление нормы матрицы Якоби
    jacobi_matrix_norm = linalg.norm(value_of_jacobi_matrix, np.inf)
    # Вывод нормы матрицы Якоби
    out_file.write('\nНорма матрицы Якоби = {0}'.format(jacobi_matrix_norm))
    # Вывод заголовка
    out_file.write('\nItr |   x           |       y       | Норма невязки     |     F1        |       F2        |\n')
    # Вычисление нормы невязки
    residual_norm = eps + 1
    # Вычисление значений с необходимой точностью
    while residual_norm >= eps:
        # Шаг итерации
        i += 1
        # Подстановка предыдущего приближения
        xi = x
        yi = y
        # Создание кортежей соответствия переменных и  их значений
        con_list_i = [(sym_x, xi), (sym_y, yi)]
        # Вычисление матрицы, обратной матрице Якоби в точке (xi, yi)
        jacobi_matrix_value = (jacobi_matrix ** -1).subs(con_list_i)
        # Вычисление нового приближения
        x = xi - jacobi_matrix_value[0, 0] * f1.subs(con_list_i) - jacobi_matrix_value[0, 1] * f2.subs(con_list_i)
        y = yi - jacobi_matrix_value[1, 0] * f1.subs(con_list_i) - jacobi_matrix_value[1, 1] * f2.subs(con_list_i)
        # Создание кортежей соответствия переменных и  их значений
        con_list = [(sym_x, x), (sym_y, y)]
        # Вычисление значения функции в точке (x,y)
        f1_value = f1.subs(con_list)
        f2_value = f2.subs(con_list)
        # Вычисление нормы невязки
        residual_norm = max(abs(x - xi), abs(y - yi))
        # Вывод результатов
        out_file.write(
            '{0} {1} {2} {3} {4} {5}\n'.format(i, x, y, residual_norm, f1_value, f2_value))


# Метод градиентного спуска
def gradient_descent_method(x, y, f1, f2, eps, out_file):
    out_file.write('\n Метод градиентного спуска\n')
    i = 0
    # TODO: Что такое k
    k = 0.5
    # ff(x,y) = (f1(x,y))^2 + (f2(x,y))ˆ2
    ff = f1 ** 2 + f2 ** 2
    # Вывод градиента
    out_file.write('Градиент\n{0}'.format(str([diff(ff, sym_x), diff(ff, sym_y)])
                                          .replace(']', '')
                                          .replace('[', '')
                                          .replace(',', '\n')))
    # Вывод заголовка
    out_file.write('\nItr |   x           |       y         |       Alpha       | Норма невязки     |     F1          '
                   ' |       F2        |       FF     | k |\n')
    # Вычисление нормы
    residual_norm = eps + 1
    while residual_norm >= eps:
        # Итерационный шаг
        i += 1
        x0 = x
        y0 = y
        # Вычисление alpha
        alpha = dihotomia(ff, -1000, 100000, x0, y0, eps)
        # Создание кортежей соответствия переменных и  их значений
        con_list0 = [(sym_x, x0), (sym_y, y0)]
        # Вычисление следующего приближения
        x = x0 - alpha * diff(ff, sym_x).subs(con_list0)
        y = y0 - alpha * diff(ff, sym_y).subs(con_list0)
        # Создание кортежей соответствия переменных и  их значений
        con_list = [(sym_x, x), (sym_y, y)]
        # Вычисление нормы
        residual_norm = max(abs(diff(f1, sym_x).subs(con_list)), abs(diff(f2, sym_y).subs(con_list)))
        # Вывод результатов
        out_file.write('{0} {1} {2} {3} {4} {5} {6} {7} {8}\n'.format(i, x, y, alpha, residual_norm, f1.subs(con_list),
                                                                      f2.subs(con_list), ff.subs(con_list), k))


# Метод половинного деления
def dihotomia(FF, a0, b0, x, y, eps) -> object:
    # Величина на которую мы отклонимся от середины отрезка
    delta = 0.5 * eps
    # Отрезок локализации минимума
    ak = a0
    bk = b0
    # Пока длина отрезка больше заданной точности
    while abs(bk - ak) >= eps:
        # Точка, отличающаяся от середины на дельту
        lk = (ak + bk - delta) / 2
        mk = (ak + bk + delta) / 2
        # Проверка в какую часть попадает точка минимума слева от разбиения или справа, выбор соответствующей точки
        if g(FF, x, y, lk) <= g(FF, x, y, mk):
            # Правая граница отрезка локализации равна mk
            bk = mk
        else:
            # Левая граница отрезка локализации равна lk
            ak = lk
    # Возврат точки минимума
    return (ak + bk) / 2


def g(f, x, y, alpha):
    dfdx = diff(f, sym_x)
    dfdy = diff(f, sym_y)
    con_list = [(sym_x, x), (sym_y, y)]
    return (x - alpha * dfdx.subs(con_list)) ** 2 + \
           (y - alpha * dfdy.subs(con_list)) ** 2
