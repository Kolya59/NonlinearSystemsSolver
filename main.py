from Solvers import *
# Допустимая погрешность
eps = 10E-4

# Ввод данных
input_file = open('test.txt', 'r')
output_file = open('out.txt', 'w')

# Первая функция в виде F1 = 0
f1 = input_file.readline().replace('\n', '')
# Вторая функция в виде F2 = 0
f2 = input_file.readline().replace('\n', '')
# Первая функция в виде y = f(x)
f11 = input_file.readline().replace('\n', '')
# Вторая функция в виде x = f(y)
f21 = input_file.readline().replace('\n', '')
# Вектор начального приближения (x,y)
x0 = float(input_file.readline().replace('\n', ''))
y0 = float(input_file.readline().replace('\n', ''))

# Вывод в файл начальных приближений
output_file.write('x0 = ' + str(x0) + '\ny0 = ' + str(y0) + '\nf1: ' + f1 + '\nf2: ' + f2)

# Метод простой итерации
simple_iteration_method(x0, y0, f11, f21, eps, output_file)

# Метод Ньютона
newton_method(x0, y0, f11, f21, eps, output_file)

# Метод градиентного спуска
gradient_descent_methd(x0, y0, f1, f2, eps, output_file)
