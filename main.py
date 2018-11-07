from sympy.parsing.sympy_parser import parse_expr

from Solvers import *

# Допустимая погрешность
eps = 10E-4

# Ввод данных
input_file = open('test.txt', 'r')
output_file = open('out.txt', 'w')

# Первая функция в виде F1 = 0
f1_str = input_file.readline().replace('\n', '')
f1 = parse_expr(f1_str)
# Вторая функция в виде F2 = 0
f2_str = input_file.readline().replace('\n', '')
f2 = parse_expr(f2_str)
# Вектор начального приближения (x,y)
x0 = float(input_file.readline().replace('\n', ''))
y0 = float(input_file.readline().replace('\n', ''))

# Вывод в файл начальных приближений
output_file.write('x0 = ' + str(x0) + '\ny0 = ' + str(y0) + '\nf1: ' + f1_str + '\nf2: ' + f2_str)

# Метод простой итерации
simple_iteration_method(x0, y0, f1, f2, eps, output_file)

# Метод Ньютона
newton_method(x0, y0, f1, f2, eps, output_file)

# Метод градиентного спуска
gradient_descent_method(x0, y0, f1, f2, eps, output_file)
