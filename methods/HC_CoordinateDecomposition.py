import numpy as np
import matplotlib.pyplot as plt


class HeatConduction2D_CoordinateDecomposition:
    def __init__(self, L=0.5, H=0.5, Nx=50, Ny=50, t_end=60, dt=0.1, Th=80, Tc=30, T0=5):
        """
        Инициализация решателя задачи теплопроводности с покоординатным расщеплением.
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

        # Инициализация поля температуры
        self.T = np.full((Ny, Nx), T0, dtype=np.float64)
        self.T[:, 0] = Th  # Левая граница
        self.T[:, -1] = Tc  # Правая граница

    def _apply_boundary_conditions(self):
        """Применение граничных условий (Дирихле)."""
        self.T[:, 0] = self.T[:, 0]  # Левая граница (Th постоянна)
        self.T[:, -1] = self.T[:, -1]  # Правая граница (Tc постоянна)
        self.T[0, 1:-1] = self.T[1, 1:-1]  # Верхняя граница (адиабатическая: dT/dy = 0)
        self.T[-1, 1:-1] = self.T[-2, 1:-1]  # Нижняя граница (адиабатическая: dT/dy = 0)

    def solve(self):
        """Решение задачи теплопроводности с использованием покоординатного расщепления."""
        r_x = self.alpha * self.dt / self.dx ** 2
        r_y = self.alpha * self.dt / self.dy ** 2

        # Проверка выполнения условия стабильности
        if r_x + r_y > 0.5:
            raise ValueError("Условие стабильности не выполнено: r_x + r_y <= 0.5")

        # Количество шагов по времени
        num_steps = int(self.t_end / self.dt)

        # Процесс решения задачи
        for step in range(num_steps):
            # Шаг по оси x
            T_x = self.T.copy()
            for i in range(1, self.Ny - 1):
                for j in range(1, self.Nx - 1):
                    T_x[i, j] = self.T[i, j] + r_x * (self.T[i, j + 1] - 2 * self.T[i, j] + self.T[i, j - 1])

            # Шаг по оси y
            T_y = T_x.copy()
            for i in range(1, self.Ny - 1):
                for j in range(1, self.Nx - 1):
                    T_y[i, j] = T_x[i, j] + r_y * (T_x[i + 1, j] - 2 * T_x[i, j] + T_x[i - 1, j])

            self.T = T_y
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
solver = HeatConduction2D_CoordinateDecomposition(L, H, Nx, Ny, t_end, dt, Th, Tc, T0)
solver.solve()
solver.plot()
