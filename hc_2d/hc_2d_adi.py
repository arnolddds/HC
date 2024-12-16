import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded


class HeatConduction2D:
    def __init__(self, L=0.5, H=0.5, Nx=50, Ny=50, dt=0.1, t_max=600):
        self.L = L  # Длина пластины
        self.H = H  # Высота пластины
        self.Nx = Nx  # Число узлов по x
        self.Ny = Ny  # Число узлов по y
        self.dx = L / (Nx - 1)  # Шаг сетки по x
        self.dy = H / (Ny - 1)  # Шаг сетки по y
        self.dt = dt  # Шаг по времени
        self.t_max = t_max  # Время решения

        # Граничные условия
        self.T = np.full((Nx, Ny), 50.0)  # Начальное поле температуры
        self.T[:, 0] = 100.0  # Левый край
        self.T[:, -1] = 0.0   # Правый край

        # Свойства материалов
        self.lambda_cu = 400  # Медь: теплопроводность
        self.rho_cu = 7800  # Плотность меди
        self.cp_cu = 460  # Теплоемкость меди

        self.inclusions = [
            {'x0': 0.1, 'y0': 0.3, 'width': 0.1, 'height': 0.1, 'lambda': 46, 'rho': 7800, 'cp': 460},  # Сталь
            {'x0': 0.3, 'y0': 0.1, 'width': 0.1, 'height': 0.2, 'lambda': 71, 'rho': 7900, 'cp': 460}   # Железо
        ]

        # Поля для параметров
        self.k = np.full((Nx, Ny), self.lambda_cu)  # Поле теплопроводности
        self.rho_cp = np.full((Nx, Ny), self.rho_cu * self.cp_cu)  # Поле ρ * c
        self.apply_inclusions()

    def apply_inclusions(self):
        """Применяем свойства материалов включений к полям."""
        for inc in self.inclusions:
            ix0 = int(inc['x0'] / self.dx)
            iy0 = int(inc['y0'] / self.dy)
            ix1 = int((inc['x0'] + inc['width']) / self.dx)
            iy1 = int((inc['y0'] + inc['height']) / self.dy)

            self.k[ix0:ix1, iy0:iy1] = inc['lambda']
            self.rho_cp[ix0:ix1, iy0:iy1] = inc['rho'] * inc['cp']

    def solve_tridiagonal(self, alpha, T, direction):
        """решение трёхдиагональной системы."""
        n = len(T)
        a = np.full(n, -alpha)  # Поддиагональ
        b = np.full(n, 1 + 2 * alpha)  # Главная диагональ
        c = np.full(n, -alpha)  # Наддиагональ
        d = T.copy()  # Правая часть

        # Граничные условия
        b[0], b[-1] = 1, 1
        c[0], a[-1] = 0, 0
        d[0], d[-1] = T[0], T[-1]

        # Решаем систему
        ab = np.zeros((3, n))
        ab[0, 1:] = c[:-1]  # Наддиагональ
        ab[1, :] = b        # Главная диагональ
        ab[2, :-1] = a[1:]  # Поддиагональ
        return solve_banded((1, 1), ab, d)

    def sweep(self):
        """Решение задачи методом продольно-поперечной прогонки."""
        T_new = self.T.copy()
        for t in range(int(self.t_max / self.dt)):
            # Продольная прогонка (x-направление)
            for j in range(1, self.Ny - 1):
                alpha_x = self.k[:, j] * self.dt / (self.rho_cp[:, j] * self.dx ** 2)
                T_new[:, j] = self.solve_tridiagonal(alpha_x, self.T[:, j], direction='x')

            # Поперечная прогонка (y-направление)
            for i in range(1, self.Nx - 1):
                alpha_y = self.k[i, :] * self.dt / (self.rho_cp[i, :] * self.dy ** 2)
                T_new[i, :] = self.solve_tridiagonal(alpha_y, T_new[i, :], direction='y')

            # Обновление температуры
            self.T = T_new.copy()
            self.T[:, 0] = 100.0  # Левый край
            self.T[:, -1] = 0.0   # Правый край
            self.T[0, :] = self.T[1, :]  # Адиабат сверху
            self.T[-1, :] = self.T[-2, :]  # Адиабат снизу

    def plot_results(self):
        """Построение изотерм с включениями."""
        X, Y = np.meshgrid(np.linspace(0, self.L, self.Nx), np.linspace(0, self.H, self.Ny))
        plt.contourf(X, Y, self.T.T, levels=20, cmap='hot')
        plt.colorbar(label='Температура (C)')
        plt.title(f'Изотермы при t = {self.t_max} с')
        plt.xlabel('x (м)')
        plt.ylabel('y (м)')

        # Включения
        for inc in self.inclusions:
            rect = plt.Rectangle((inc['x0'], inc['y0']), inc['width'], inc['height'], color='blue', alpha=0.5)
            plt.gca().add_patch(rect)
        plt.show()


if __name__ == "__main__":
    solver = HeatConduction2D(Nx=100, Ny=100, dt=0.01, t_max=300)
    solver.sweep()
    solver.plot_results()
