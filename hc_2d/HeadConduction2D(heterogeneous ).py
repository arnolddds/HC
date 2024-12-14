import numpy as np
import matplotlib.pyplot as plt

class HeatConductionHeterogeneous2D:
    def __init__(self, L=0.5, H=0.5, Nx=50, Ny=50, t_end=60, dt=0.1,
                 nx1=10, nx2=10, nx3=10, nx4=10, ny1=10, ny2=10, ny3=10, ny4=10):
        """
        Инициализация решателя двумерной задачи теплопроводности для неоднородного тела.
        :param L: Длина пластины (метры, по оси X)
        :param H: Высота пластины (метры, по оси Y)
        :param Nx: Число узлов по X
        :param Ny: Число узлов по Y
        :param t_end: Время моделирования (секунды)
        :param dt: Шаг по времени (секунды)
        :param nx1, nx2, nx3, nx4: Число промежутков для участков по X
        :param ny1, ny2, ny3, ny4: Число промежутков для участков по Y
        """
        self.L, self.H = L, H
        self.Nx, self.Ny = Nx, Ny
        self.dx, self.dy = L / (Nx - 1), H / (Ny - 1)
        self.dt, self.t_end = dt, t_end

        # Число промежутков для включений
        self.nx1, self.nx2, self.nx3, self.nx4 = nx1, nx2, nx3, nx4
        self.ny1, self.ny2, self.ny3, self.ny4 = ny1, ny2, ny3, ny4

        # Свойства материалов (1: медь, 2: сталь, 3: железо)
        self.material_properties = {
            1: {'lambda': 401, 'rho': 8960, 'c': 385},  # Медь
            2: {'lambda': 46, 'rho': 7800, 'c': 460},   # Сталь
            3: {'lambda': 71, 'rho': 7900, 'c': 460},   # Железо
        }

        # Инициализация температурного поля
        self.T = np.full((Ny, Nx), 50.0, dtype=np.float64)  # Начальная температура T0 = 50°C
        self.T[:, 0] = 100.0  # Левая граница (Th = 100°C)
        self.T[:, -1] = 0.0   # Правая граница (Tc = 0°C)

        # Определение сетки материалов
        self.material_grid = np.ones((Ny, Nx), dtype=int)  # По умолчанию медь (1)
        self._add_inclusions()

    def _add_inclusions(self):
        """
        Добавление включений из стали и железа в материал.
        """
        # Расчет узлов для включений по X
        x1 = self.nx1
        x2 = x1 + self.nx2
        x3 = x2 + self.nx3
        x4 = x3 + self.nx4

        # Расчет узлов для включений по Y
        y1 = self.ny1
        y2 = y1 + self.ny2
        y3 = y2 + self.ny3
        y4 = y3 + self.ny4

        # Включение из стали (материал 2)
        self.material_grid[y1:y2, x1:x2] = 2

        # Включение из железа (материал 3)
        self.material_grid[y3:y4, x3:x4] = 3

    def _apply_boundary_conditions(self):
        """
        Применение граничных условий Дирихле и Неймана.
        """
        self.T[:, 0] = 100.0  # Левая граница (Th = 100°C)
        self.T[:, -1] = 0.0   # Правая граница (Tc = 0°C)
        self.T[0, 1:-1] = self.T[1, 1:-1]  # Верхняя граница (адиабатическая: dT/dy = 0)
        self.T[-1, 1:-1] = self.T[-2, 1:-1]  # Нижняя граница (адиабатическая: dT/dy = 0)

    def solve(self):
        """
        Решение задачи теплопроводности методом явных конечных разностей.
        """
        num_steps = int(self.t_end / self.dt)

        for step in range(num_steps):
            T_new = self.T.copy()

            for i in range(1, self.Ny - 1):
                for j in range(1, self.Nx - 1):
                    material = self.material_grid[i, j]
                    props = self.material_properties[material]

                    # Коэффициент температуропроводности: alpha = lambda / (rho * c)
                    alpha = props['lambda'] / (props['rho'] * props['c'])

                    # Обновление температуры методом явной схемы
                    T_new[i, j] = (
                        self.T[i, j]
                        + alpha * self.dt * (
                            (self.T[i, j + 1] - 2 * self.T[i, j] + self.T[i, j - 1]) / self.dx**2
                            + (self.T[i + 1, j] - 2 * self.T[i, j] + self.T[i - 1, j]) / self.dy**2
                        )
                    )

            self.T = T_new
            self._apply_boundary_conditions()

        return self.T

    def plot(self):
        """
        Построение распределения температуры и отображение включений.
        """
        x = np.linspace(0, self.L, self.Nx)
        y = np.linspace(0, self.H, self.Ny)
        X, Y = np.meshgrid(x, y)

        plt.figure(figsize=(10, 6))
        # Построение изотерм температуры
        cp = plt.contourf(X, Y, self.T, levels=20, cmap="coolwarm")
        plt.colorbar(cp, label="Температура (°C)")

        # Наложение включений
        for i in range(self.Ny):
            for j in range(self.Nx):
                if self.material_grid[i, j] == 2:  # Сталь
                    plt.scatter(x[j], y[i], color='blue', s=1, label='Сталь' if i == j == 1 else "")
                elif self.material_grid[i, j] == 3:  # Железо
                    plt.scatter(x[j], y[i], color='green', s=1, label='Железо' if i == j == 1 else "")

        # Настройка графика
        plt.title(f"Распределение температуры при t = {self.t_end} с")
        plt.xlabel("X (м)")
        plt.ylabel("Y (м)")
        plt.legend(loc="upper right")
        plt.show()

# Запуск программы
if __name__ == "__main__":
    solver = HeatConductionHeterogeneous2D(Nx=50, Ny=50, t_end=120, dt=0.1,
                                           nx1=10, nx2=10, nx3=10, nx4=10,
                                           ny1=10, ny2=10, ny3=10, ny4=10)
    solver.solve()
    solver.plot()
