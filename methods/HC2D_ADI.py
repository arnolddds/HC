import numpy as np
import matplotlib.pyplot as plt


class HeatConduction2D_ADI:
    def __init__(self, L=0.5, H=0.5, Nx=50, Ny=50, t_end=60, dt=0.1, Th=80, Tc=30, T0=5):
        """
        Инициализация решателя задачи теплопроводности методом продольно-поперечной прогонки.
        :param L: Длина пластины (по оси x, в метрах)
        :param H: Высота пластины (по оси y, в метрах)
        :param Nx: Количество узлов сетки по оси x
        :param Ny: Количество узлов сетки по оси y
        :param t_end: Время окончания симуляции (в секундах)
        :param dt: Временной шаг (в секундах)
        :param Th: Температура на левой границе (°C)
        :param Tc: Температура на правой границе (°C)
        :param T0: Начальная температура по всей пластине (°C)
        """
        self.L, self.H = L, H
        self.Nx, self.Ny = Nx, Ny
        self.dx, self.dy = L / (Nx - 1), H / (Ny - 1)
        self.dt, self.t_end = dt, t_end

        # Термальные свойства меди
        self.alpha = 401 / (8960 * 385)  # Теплопроводность, альфа = lambda / (rho * c)

        # Параметры для метода прогонки
        self.r_x = self.alpha * self.dt / (2 * self.dx**2)
        self.r_y = self.alpha * self.dt / (2 * self.dy**2)

        # Инициализация поля температуры
        self.T = np.full((Ny, Nx), T0, dtype=np.float64)
        self.T[:, 0] = Th  # Левая граница
        self.T[:, -1] = Tc  # Правая граница

    def _apply_boundary_conditions(self):
        """Применение граничных условий."""
        self.T[:, 0] = self.T[:, 0]  # Левая граница
        self.T[:, -1] = self.T[:, -1]  # Правая граница
        self.T[0, :] = self.T[1, :]  # Верхняя граница (адиабатическая)
        self.T[-1, :] = self.T[-2, :]  # Нижняя граница (адиабатическая)

    def _tridiagonal_solver(self, a, b, c, d):
        """Решение системы с трёхдиагональной матрицей методом прогонки."""
        n = len(d)
        c_star = np.zeros(n - 1)
        d_star = np.zeros(n)

        # Прямая прогонка
        c_star[0] = c[0] / b[0]
        d_star[0] = d[0] / b[0]
        for i in range(1, n):
            denom = b[i] - a[i - 1] * c_star[i - 1]
            if i < n - 1:
                c_star[i] = c[i] / denom
            d_star[i] = (d[i] - a[i - 1] * d_star[i - 1]) / denom

        # Обратная прогонка
        x = np.zeros(n)
        x[-1] = d_star[-1]
        for i in range(n - 2, -1, -1):
            x[i] = d_star[i] - c_star[i] * x[i + 1]

        return x

    def solve(self):
        """Решение задачи методом продольно-поперечной прогонки (ADI)."""
        num_steps = int(self.t_end / self.dt)
        for step in range(num_steps):
            # Первый полушаг: решение по x (фиксированный y)
            T_half = self.T.copy()
            for i in range(1, self.Ny - 1):
                a = -self.r_x * np.ones(self.Nx - 2)
                b = (1 + 2 * self.r_x) * np.ones(self.Nx - 2)
                c = -self.r_x * np.ones(self.Nx - 2)
                d = (
                    self.T[i, 1:-1]
                    + self.r_y * (self.T[i + 1, 1:-1] - 2 * self.T[i, 1:-1] + self.T[i - 1, 1:-1])
                )
                T_half[i, 1:-1] = self._tridiagonal_solver(a, b, c, d)

            # Второй полушаг: решение по y (фиксированный x)
            for j in range(1, self.Nx - 1):
                a = -self.r_y * np.ones(self.Ny - 2)
                b = (1 + 2 * self.r_y) * np.ones(self.Ny - 2)
                c = -self.r_y * np.ones(self.Ny - 2)
                d = (
                    T_half[1:-1, j]
                    + self.r_x * (T_half[1:-1, j + 1] - 2 * T_half[1:-1, j] + T_half[1:-1, j - 1])
                )
                self.T[1:-1, j] = self._tridiagonal_solver(a, b, c, d)

            self._apply_boundary_conditions()

        return self.T

    def plot(self):
        """Построение графика распределения температуры в виде контурного графика."""
        x = np.linspace(0, self.L, self.Nx)
        y = np.linspace(0, self.H, self.Ny)
        X, Y = np.meshgrid(x, y)

        plt.figure(figsize=(8, 6))
        cp = plt.contourf(X, Y, self.T, levels=20, cmap="coolwarm")
        plt.colorbar(cp, label="Температура (°C)")
        plt.title(f"Распределение температуры при t = {self.t_end} с")
        plt.xlabel("X (м)")
        plt.ylabel("Y (м)")
        plt.show()


# Параметры задачи
L, H = 0.5, 0.5  # Размеры пластины (в метрах)
Nx, Ny = 51, 51  # Количество узлов сетки
Th, Tc, T0 = 80, 30, 5  # Граничные и начальные условия
t_end, dt = 60, 0.1  # Время симуляции и временной шаг

# Создание и решение задачи
solver = HeatConduction2D_ADI(L, H, Nx, Ny, t_end, dt, Th, Tc, T0)
solver.solve()
solver.plot()
