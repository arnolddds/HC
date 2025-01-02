import numpy as np
import matplotlib.pyplot as plt


class HeatTransferLongitudinalTransverse:
    def __init__(self, L, H, Nx, Ny, lambda_val, rho, c, T0, Th, Tc):
        """
        # Инициализация задачи теплопроводности методом продольно-поперечной прогонки
        """
        # Геометрические параметры
        self.L = L
        self.H = H
        self.Nx = Nx
        self.Ny = Ny

        # Шаги сетки
        self.hx = L / (Nx - 1)
        self.hy = H / (Ny - 1)

        # Физические параметры
        self.lambda_val = lambda_val
        self.rho = rho
        self.c = c

        # Температурные параметры
        self.T0 = T0
        self.Th = Th
        self.Tc = Tc

        # Инициализация полей температуры
        self.T = np.ones((Nx, Ny)) * T0
        self.T1D = np.ones(Nx) * T0

        # Задание граничных условий
        self.T[0, :] = Th
        self.T[-1, :] = Tc
        self.T1D[0] = Th
        self.T1D[-1] = Tc

        # Для хранения истории решений
        self.history_1D = []
        self.history_2D = []

    def thomas_algorithm(self, a, b, c, d):
        """
        # Метод прогонки для решения СЛАУ с трехдиагональной матрицей
        """
        n = len(d)
        alpha = np.zeros(n)
        beta = np.zeros(n)
        x = np.zeros(n)

        alpha[0] = c[0] / b[0]
        beta[0] = d[0] / b[0]

        for i in range(1, n):
            denom = b[i] - a[i] * alpha[i - 1]
            alpha[i] = c[i] / denom
            beta[i] = (d[i] - a[i] * beta[i - 1]) / denom

        x[-1] = beta[-1]
        for i in range(n - 2, -1, -1):
            x[i] = beta[i] - alpha[i] * x[i + 1]

        return x

    def solve_timestep(self, tau):
        """
        # Решение одного временного шага методом продольно-поперечной прогонки
        """

        T_new = np.copy(self.T)

        # Решение одномерной задачи
        a1D = np.full(self.Nx, self.lambda_val / self.hx ** 2)
        b1D = np.full(self.Nx, -2 * self.lambda_val / self.hx ** 2 - self.rho * self.c / tau)
        c1D = np.full(self.Nx, self.lambda_val / self.hx ** 2)
        d1D = -self.rho * self.c * self.T1D / tau

        # Граничные условия для 1D
        b1D[0] = 1
        c1D[0] = 0
        d1D[0] = self.Th

        a1D[-1] = 0
        b1D[-1] = 1
        d1D[-1] = self.Tc

        self.T1D = self.thomas_algorithm(a1D, b1D, c1D, d1D)

        # Прогонка вдоль строк (x-направление)
        for j in range(1, self.Ny - 1):
            # Формируем коэффициенты для прогонки
            ax = np.full(self.Nx, self.lambda_val / self.hx ** 2)
            bx = np.full(self.Nx,
                         -2 * self.lambda_val / self.hx ** 2 - 2 * self.lambda_val / self.hy ** 2 - self.rho * self.c / tau)
            cx = np.full(self.Nx, self.lambda_val / self.hx ** 2)
            dx = -self.rho * self.c * self.T[:, j] / tau - self.lambda_val * (
                        self.T[:, j + 1] + self.T[:, j - 1]) / self.hy ** 2

            # ГУ
            bx[0] = 1
            cx[0] = 0
            dx[0] = self.Th

            ax[-1] = 0
            bx[-1] = 1
            dx[-1] = self.Tc


            T_new[:, j] = self.thomas_algorithm(ax, bx, cx, dx)

        # Прогонка вдоль столбцов (y-направление)
        for i in range(1, self.Nx - 1):

            ay = np.full(self.Ny, self.lambda_val / self.hy ** 2)
            by = np.full(self.Ny,
                         -2 * self.lambda_val / self.hy ** 2 - 2 * self.lambda_val / self.hx ** 2 - self.rho * self.c / tau)
            cy = np.full(self.Ny, self.lambda_val / self.hy ** 2)
            dy = -self.rho * self.c * T_new[i, :] / tau - self.lambda_val * (
                        T_new[i + 1, :] + T_new[i - 1, :]) / self.hx ** 2


            ay[0] = -1
            by[0] = 1
            cy[0] = 0
            dy[0] = 0

            ay[-1] = -1
            by[-1] = 1
            cy[-1] = 0
            dy[-1] = 0

            # Решаем систему методом прогонки
            self.T[i, :] = self.thomas_algorithm(ay, by, cy, dy)

        # Сохраняем граничные условия
        self.T[0, :] = self.Th
        self.T[-1, :] = self.Tc

    def solve(self, t_end):
        """
        # Решение задачи до заданного момента времени
        """
        t = 0
        tau = t_end / 1000 #шаг

        while t < t_end:
            self.solve_timestep(tau)
            t += tau


            if abs(t - 10) < tau or abs(t - 30) < tau or abs(t - 60) < tau:
                self.history_1D.append(np.copy(self.T1D))
                self.history_2D.append(np.copy(self.T))

    def plot_results(self):
        """
        # Визуализация результатов
        """
        times = [100, 300, 600]
        fig = plt.figure(figsize=(15, 10))

        for idx, t in enumerate(times):

            plt.subplot(2, 3, idx + 1)
            x = np.linspace(0, self.L, self.Nx)
            plt.plot(x, self.history_1D[idx], 'b-', label='1D решение')
            plt.plot(x, self.history_2D[idx][:, self.Ny // 2], 'r--',
                     label='2D решение')
            plt.title(f't = {t} с')
            plt.xlabel('x (м)')
            plt.ylabel('Температура (°C)')
            plt.legend()
            plt.grid(True)

            # Двумерное распределение температуры
            plt.subplot(2, 3, idx + 4)
            y = np.linspace(0, self.H, self.Ny)
            X, Y = np.meshgrid(x, y)
            plt.contourf(X, Y, self.history_2D[idx].T, levels=20, cmap='jet')
            plt.colorbar(label='Температура (°C)')
            plt.xlabel('x (м)')
            plt.ylabel('y (м)')

        plt.tight_layout()
        plt.show()


# Пример использования
if __name__ == "__main__":
    # Параметры задачи
    L = 0.1 # длина пластины
    H = 0.1 # высота пластины
    Nx = Ny = 50  # количество узлов сетки

    # Свойства пластины
    lambda_val = 75  # теплопроводность 384
    rho = 7800  # плотность 8800
    c = 439  # теплоемкость 381

    # Температурные условия
    T0 = 5  # начальная температура T0 = 5
    Th = 80  # температура слева Th = 80
    Tc = 30  # температура справа Tc = 30


    solver = HeatTransferLongitudinalTransverse(L, H, Nx, Ny, lambda_val, rho, c, T0, Th, Tc)
    solver.solve(t_end=600)
    solver.plot_results()