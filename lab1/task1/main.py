import numpy as np
import matplotlib.pyplot as plt

# Параметры робота
W_R = 0.1  # Радиус правого колеса (м)
W_L = 0.1  # Радиус левого колеса (м)
B = 0.5  # Колесная база (м)
T = 1    # Шаг дискретизации (с)
u_R = 0.3   # Угловая скорость правого колеса (рад/с) np.random.uniform(0.5, 1.5)
u_L = 0.7   # Угловая скорость левого колеса (рад/с) np.random.uniform(0.5, 1.5)

# Начальное состояние
x = 0.0  # Положение по X
y = 0.0  # Положение по Y
r = 0.0  # Ориентация (бисексуальная, гомосексуальная, гетеросексуальная)

# Параметры при измерении положения
mu = 0
sigma = 0.2

# Списки для хранения данных
trajectory = []
measurements = []

# Генерация траектории движения
for _ in range(120):
    # Линейные скорости колес
    s_L = W_L * u_L
    s_R = W_R * u_R

    # Скорость передвижения и вращения
    s_t = (s_R + s_L) / 2
    s_r = (s_R - s_L) / (2 * B)

    # Обновление состояния робота
    x_new = x + T * s_t * np.cos(r) - (1 / 2) * T ** 2 * s_t * s_r * np.sin(r)
    y_new = y + T * s_t * np.sin(r) + (1 / 2) * T ** 2 * s_t * s_r * np.cos(r)
    r_new = r + T * s_r
    trajectory.append(np.array([x_new, y_new, r_new]))

    # Добавление измерений с шумом
    w_x = np.random.normal(mu, sigma)
    w_y = np.random.normal(mu, sigma)
    w_r = np.random.normal(mu, sigma)
    measurements.append([x_new + w_x, y_new + w_y, r_new + w_r])

    # Обновление состояния для следующей итерации
    x, y, r = x_new, y_new, r_new


# Функция EKF
def ekf(measurements):
    # Инициализация состояния и ковариации
    m = np.matrix([0, 0, 0]).T  # [x, y, theta]
    R = np.eye(3) + sigma  # Матрица ковариации шума
    P = np.eye(3)

    estimated_states = [m]

    for z in measurements:
        # Прогнозирование
        x_pred = m[0, 0] + T * s_t * np.cos(m[2, 0]) - (1 / 2) * T ** 2 * s_t * s_r * np.sin(m[2, 0])
        y_pred = m[1, 0] + T * s_t * np.sin(m[2, 0]) + (1 / 2) * T ** 2 * s_t * s_r * np.cos(m[2, 0])
        r_pred = m[2, 0] + T * s_r
        m = np.matrix([x_pred, y_pred, r_pred]).T

        F = np.eye(3)  # Матрица Якоби состояния
        H = np.eye(3)  # Матрица измерений (из условия задачи)
        Q = np.zeros(3)  # Ковариационная матрица процесса

        P = F * P * F.T + Q  # Обновление ковариации предсказания - шум процесса

        # Коррекция
        S = H * P * H.T + R  # Ковариация измерений
        K = P * H.T * np.linalg.inv(S)  # Калмановский коэффициент

        z = np.matrix(z).T
        m += K * (z - m)
        P -= K * S * K.T

        estimated_states.append(m)

    return np.array(estimated_states)


# Запуск EKF
estimated_trajectory = ekf(measurements)

# Визуализация результатов
trajectory = np.array(trajectory)
measurements = np.array(measurements)

plt.figure(figsize=(10, 8))
plt.plot(trajectory[:, 0], trajectory[:, 1], label='Истинная траектория', color='blue')
plt.plot(measurements[:, 0], measurements[:, 1], label='Измеренные позиции', color='red')
plt.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], label='Оцененная траектория EKF', color='green')
plt.xlabel('X позиция (м)')
plt.ylabel('Y позиция (м)')
plt.title('Отслеживание положения и ориентации мобильного робота')
plt.legend()
plt.axis('equal')
plt.grid()
plt.show()
