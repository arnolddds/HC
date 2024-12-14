import numpy as np
import matplotlib.pyplot as plt

class TwoLayerPlate:
    def __init__(self, N1, N2, t_end, L, lambda1, lambda2, ro1, ro2, c1, c2, T0, Tl, Tr):
        self.N1 = N1
        self.N2 = N2
        self.N = N1 + N2
        self.t_end = t_end
        self.L = L
        self.h = self.L / self.N  # Шаг по пространству
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.ro1 = ro1
        self.ro2 = ro2
        self.c1 = c1
        self.c2 = c2
        self.T0 = T0
        self.Tl = Tl
        self.Tr = Tr
        self.dt = 0.01  # Шаг по времени
        self.T = np.full(self.N, self.T0)
        self.T[0] = self.Tl
        self.T[-1] = self.Tr
        self.T_history = []
        self.time_points = {}  # Словарь для сохранения температур в указанные моменты времени

    def solve(self, time_points_to_save):
        for step in range(int(self.t_end / self.dt)):
            current_time = step * self.dt
            # Коэффициенты для системы
            a = np.zeros(self.N)
            b = np.zeros(self.N)
            c = np.zeros(self.N)
            d = np.zeros(self.N)

            # Граничные условия
            b[0] = 1
            d[0] = self.Tl

            b[-1] = 1
            d[-1] = self.Tr

            for i in range(1, self.N - 1):
                if i < self.N1:  # Первый слой
                    lambdai_plus_half = self.lambda1
                    lambdai_minus_half = self.lambda1
                    ro_c = self.ro1 * self.c1
                elif i == self.N1:  # Граница между слоями
                    lambdai_plus_half = self.lambda2  # После границы
                    lambdai_minus_half = self.lambda1  # До границы
                    ro_c = (self.ro1 * self.c1 + self.ro2 * self.c2) / 2  # Усреднённая теплоёмкость
                else:  # Второй слой
                    lambdai_plus_half = self.lambda2
                    lambdai_minus_half = self.lambda2
                    ro_c = self.ro2 * self.c2

                # Основное уравнение для каждого узла
                a[i] = lambdai_minus_half / (self.h ** 2)
                c[i] = lambdai_plus_half / (self.h ** 2)
                b[i] = - (a[i] + c[i] + ro_c / self.dt)
                d[i] = - self.T[i] * ro_c / self.dt

            # Метод прогонки для решения системы уравнений
            alpha = np.zeros(self.N)
            beta = np.zeros(self.N)
            alpha[1] = -c[0] / b[0]
            beta[1] = d[0] / b[0]

            for i in range(1, self.N - 1):
                denominator = a[i] * alpha[i] + b[i]
                alpha[i + 1] = -c[i] / denominator
                beta[i + 1] = (d[i] - a[i] * beta[i]) / denominator

            T_new = np.zeros_like(self.T)
            T_new[-1] = (d[-1] - a[-1] * beta[-1]) / (b[-1] + a[-1] * alpha[-1])

            for i in range(self.N - 2, -1, -1):
                T_new[i] = alpha[i + 1] * T_new[i + 1] + beta[i + 1]

            self.T = T_new
            self.T_history.append(self.T.copy())

            # Сохранение температуры для указанных временных моментов
            if current_time in time_points_to_save:
                self.time_points[current_time] = self.T.copy()

    def plot(self):
        x = np.linspace(0, self.L, self.N)
        for time, temp in self.time_points.items():
            plt.plot(x, temp, label=f't = {time:.0f} сек')
        plt.xlabel('Длина пластины (м)')
        plt.ylabel('Температура (°C)')
        plt.title('Распределение температуры по двум слоям пластины')
        plt.grid(True)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    N1 = 50  # Количество ячеек в первом слое
    N2 = 50  # Количество ячеек во втором слое
    t_end = 800  # Общее время
    L = 0.4  # Длина пластины
    lambda1 = 46.0  # Теплопроводность первого слоя
    lambda2 = 384.0  # Теплопроводность второго слоя
    ro1 = 7800.0  # Плотность первого слоя
    ro2 = 8800.0  # Плотность второго слоя
    c1 = 460.0  # Удельная теплоёмкость первого слоя
    c2 = 381.0  # Удельная теплоёмкость второго слоя
    K = 273.0  # Константа для приведения к абсолютной температуре
    T0 = 10 + K  # Начальная температура
    Tl = 100 + K  # Температура слева
    Tr = 50 + K  # Температура справа

    # Указанные моменты времени
    time_points = [30, 180, 600]

    two_layer_plate = TwoLayerPlate(N1, N2, t_end, L, lambda1, lambda2, ro1, ro2, c1, c2, T0, Tl, Tr)
    two_layer_plate.solve(time_points_to_save=time_points)
    two_layer_plate.plot()
