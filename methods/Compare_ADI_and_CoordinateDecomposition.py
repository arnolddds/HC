import numpy as np
import matplotlib.pyplot as plt
import time

class HeatConduction2D_ADI:
    def __init__(self, L=0.5, H=0.5, Nx=50, Ny=50, t_end=60, dt=0.1, Th=80, Tc=30, T0=5):
        self.L, self.H = L, H  # Размеры области
        self.Nx, self.Ny = Nx, Ny  # Число ячеек по X и Y
        self.dx, self.dy = L / (Nx - 1), H / (Ny - 1)  # Шаги по пространству
        self.dt, self.t_end = dt, t_end  # Шаг по времени и конечное время
        self.alpha = 401 / (8960 * 385)  # Термальная диффузия
        self.T = np.full((Ny, Nx), T0, dtype=np.float64)  # Начальная температура
        self.T[:, 0] = Th  # Левая граница
        self.T[:, -1] = Tc  # Правая граница

    def solve(self):
        r_x = self.alpha * self.dt / self.dx**2  # Безразмерный параметр для X
        r_y = self.alpha * self.dt / self.dy**2  # Безразмерный параметр для Y
        if r_x > 0.5 or r_y > 0.5:  # Условие устойчивости
            raise ValueError("Условие устойчивости для ADI нарушено")

        num_steps = int(self.t_end / self.dt)  # Число шагов по времени
        for _ in range(num_steps):
            # Шаг 1: Решение по оси X
            for j in range(1, self.Ny - 1):
                A = np.diag((1 + 2 * r_x) * np.ones(self.Nx - 2))  # Матрица A
                B = np.diag(-r_x * np.ones(self.Nx - 3), k=1)  # Матрица B
                C = np.diag(-r_x * np.ones(self.Nx - 3), k=-1)  # Матрица C
                M = A + B + C  # Полная матрица

                b = self.T[j, 1:-1] + r_y * (self.T[j - 1, 1:-1] - 2 * self.T[j, 1:-1] + self.T[j + 1, 1:-1])  # Вектор правой части
                self.T[j, 1:-1] = np.linalg.solve(M, b)  # Решение для текущего слоя

            # Шаг 2: Решение по оси Y
            for i in range(1, self.Nx - 1):
                A = np.diag((1 + 2 * r_y) * np.ones(self.Ny - 2))  # Матрица A
                B = np.diag(-r_y * np.ones(self.Ny - 3), k=1)  # Матрица B
                C = np.diag(-r_y * np.ones(self.Ny - 3), k=-1)  # Матрица C
                M = A + B + C  # Полная матрица

                b = self.T[1:-1, i] + r_x * (self.T[1:-1, i - 1] - 2 * self.T[1:-1, i] + self.T[1:-1, i + 1])  # Вектор правой части
                self.T[1:-1, i] = np.linalg.solve(M, b)  # Решение для текущего слоя

        return self.T


class HeatConduction2D_Splitting:
    def __init__(self, L=0.5, H=0.5, Nx=50, Ny=50, t_end=60, dt=0.1, Th=80, Tc=30, T0=5):
        self.L, self.H = L, H  # Размеры области
        self.Nx, self.Ny = Nx, Ny  # Число ячеек по X и Y
        self.dx, self.dy = L / (Nx - 1), H / (Ny - 1)  # Шаги по пространству
        self.dt, self.t_end = dt, t_end  # Шаг по времени и конечное время
        self.alpha = 401 / (8960 * 385)  # Термальная диффузия
        self.T = np.full((Ny, Nx), T0, dtype=np.float64)  # Начальная температура
        self.T[:, 0] = Th  # Левая граница
        self.T[:, -1] = Tc  # Правая граница

    def solve(self):
        r_x = self.alpha * self.dt / self.dx**2  # Безразмерный параметр для X
        r_y = self.alpha * self.dt / self.dy**2  # Безразмерный параметр для Y
        if r_x + r_y > 0.5:  # Условие устойчивости
            raise ValueError("Условие устойчивости для метода разложения нарушено")

        num_steps = int(self.t_end / self.dt)  # Число шагов по времени
        for _ in range(num_steps):
            T_new = self.T.copy()  # Копия текущего состояния
            # Шаг 1: Решение по оси X
            for i in range(1, self.Ny - 1):
                for j in range(1, self.Nx - 1):
                    T_new[i, j] = self.T[i, j] + r_x * (self.T[i, j + 1] - 2 * self.T[i, j] + self.T[i, j - 1])  # Обновление температуры

            self.T = T_new.copy()  # Обновляем температуру после расчета по X

            # Шаг 2: Решение по оси Y
            for i in range(1, self.Ny - 1):
                for j in range(1, self.Nx - 1):
                    T_new[i, j] = self.T[i, j] + r_y * (self.T[i + 1, j] - 2 * self.T[i, j] + self.T[i - 1, j])  # Обновление температуры

            self.T = T_new  # Обновляем температуру после расчета по Y

        return self.T


# Параметры задачи
L, H = 0.5, 0.5
Nx, Ny = 51, 51
Th, Tc, T0 = 80, 30, 5
t_end, dt = 60, 0.1

# Решение с методом ADI
solver_adi = HeatConduction2D_ADI(L, H, Nx, Ny, t_end, dt, Th, Tc, T0)
start_adi = time.time()
T_adi = solver_adi.solve()
time_adi = time.time() - start_adi

# Решение с методом разложения
solver_split = HeatConduction2D_Splitting(L, H, Nx, Ny, t_end, dt, Th, Tc, T0)
start_split = time.time()
T_split = solver_split.solve()
time_split = time.time() - start_split

# Графики результатов
x = np.linspace(0, L, Nx)
y = np.linspace(0, H, Ny)
X, Y = np.meshgrid(x, y)

plt.figure(figsize=(12, 6))

# График для метода ADI
plt.subplot(1, 2, 1)
plt.contourf(X, Y, T_adi, levels=20, cmap="coolwarm")
plt.colorbar(label="Температура (°C)")
plt.title(f"Решение методом ADI (Время: {time_adi:.2f} с)")
plt.xlabel("X (м)")
plt.ylabel("Y (м)")

# График для метода разложения
plt.subplot(1, 2, 2)
plt.contourf(X, Y, T_split, levels=20, cmap="coolwarm")
plt.colorbar(label="Температура (°C)")
plt.title(f"Решение методом разложения (Время: {time_split:.2f} с)")
plt.xlabel("X (м)")
plt.ylabel("Y (м)")

plt.tight_layout()
plt.show()
