import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
import math

mu = 0
sigma = 0.2


def load_data():
    landmarks = {}
    with open('../data_files/landmarks.dat') as f:
        for line in f:
            id_, x, y = map(int, line.strip().split())
            landmarks[int(id_)] = (x, y)

    sensor_data = {}
    current_odometry = None
    with open('../data_files/sensor_data_ekf.dat') as f:
        for line in f:
            parts = list(line.strip().split())
            if parts[0] == "ODOMETRY":
                current_odometry = tuple(map(float, parts[1:]))
                if current_odometry not in sensor_data:
                    sensor_data[current_odometry] = []

            elif parts[0] == "SENSOR":
                if current_odometry is not None:
                    # Обрабатываем данные сенсора
                    sensor_values = (int(parts[1]), list(map(float, parts[2:])))
                    sensor_data[current_odometry].append(sensor_values)

    return landmarks, sensor_data


def draw(ekf_states, ukf_states, landmarks):
    plt.figure(figsize=(12, 8))
    landmark_coords = np.array(list(landmarks.values()))
    plt.scatter(landmark_coords[:, 0], landmark_coords[:, 1],
                marker='*', c='red', s=100, label='Landmarks')
    plt.plot(ekf_states[:, 0], ekf_states[:, 1], label='EKF')
    plt.plot(ukf_states[:, 0], ukf_states[:, 1], '--', label='UKF')
    plt.legend()
    plt.title('Сравнение траекторий EKF и UKF')
    plt.xlabel('X координата')
    plt.ylabel('Y координата')
    plt.grid(True)
    plt.show()


def ekf(state, P, odometry, landmarks, measurements):
    # Прогнозирование
    dr1, dt, dr2 = odometry

    x, y, theta = state
    new_x = x + dt * np.cos(theta + dr1)
    new_y = y + dt * np.sin(theta + dr1)
    new_theta = theta + dr1 + dr2
    state = np.array([new_x, new_y, new_theta])

    F = np.array([
        [1, 0, -dt * np.sin(theta + dr1)],
        [0, 1, dt * np.cos(theta + dr1)],
        [0, 0, 1]
    ])                     # Матрица Якоби модели движения
    Q = np.eye(3) * sigma  # Ковариационная матрица процесса
    P = F @ P @ F.T + Q    # Обновление ковариации предсказания - шум процесса

    # Коррекция
    if measurements:
        num_meas = len(measurements)  # количество сенсоров
        z_pred = np.zeros(num_meas)   # ожидаемые измерения
        z_meas = np.zeros(num_meas)   # реальные измерения
        H = np.zeros((num_meas, 3))   # матрица Якоби измерений

        for i, (landmark_id, z) in enumerate(measurements):
            lx, ly = landmarks[landmark_id]  # координаты сенсора
            dx = state[0] - lx
            dy = state[1] - ly

            z_pred[i] = np.sqrt(dx**2 + dy**2)              # ожидаемое расстояние
            z_meas[i] = z[0]                                # реальное расстояние
            H[i, :] = [dx / z_pred[i], dy / z_pred[i], 0]   # матрица Якоби модели измерения из подсчитанных производных

        R = np.eye(num_meas) * sigma
        S = H @ P @ H.T + R             # Ковариация измерений
        K = P @ H.T @ np.linalg.inv(S)  # Калмановский коэффициент

        state += (K @ (z_meas - z_pred))
        P -= K @ S @ K.T

    return state, P


def ukf_compute_sigma_points(state, P, n):
    sigma_points = np.zeros((2 * n + 1, n))
    sigma_points[0] = state     # центральная точка
    C = np.linalg.cholesky(P)   # разложение Холецкого

    for i in range(n):
        sigma_points[i + 1] = state + C[i]          # положительные отклонения
        sigma_points[n + i + 1] = state - C[i]      # отрицательное отклонение
    return sigma_points


def ukf(state, P, odometry, landmarks, measurements):
    # Прогнозирование
    n = 3
    dr1, dt, dr2 = odometry

    # веса
    lambd = 1
    W = [1 / 2 / (n + lambd) for _ in range(2 * n + 1)]
    W[0] = lambd / (n + lambd)

    # сигма-точки
    sigma_points = ukf_compute_sigma_points(state, P, n)
    for i in range(2 * n + 1):
        x, y, theta = sigma_points[i]
        sigma_points[i] = [
            x + dt * np.cos(theta + dr1),
            y + dt * np.sin(theta + dr1),
            theta + dr1 + dr2
        ]

    # состояние и ковариация
    state = np.dot(W, sigma_points)  # взвешенное среднее сигма-точек
    delta = sigma_points - state
    Q = np.eye(3) * sigma            # ковариационная матрица процесса
    P = (delta.T * W) @ delta + Q

    # Коррекция
    if measurements:
        sigma_points = ukf_compute_sigma_points(state, P, n)
        num_meas = len(measurements)

        # сигма-точек -> расстояния
        z_sigma = np.zeros((2 * n + 1, num_meas))  # расстояния между сигма-точками и сенсорами
        z_meas = np.zeros(num_meas)                # реальные расстояния до сенсоров
        for i, (landmark_id, z) in enumerate(measurements):
            lx, ly = landmarks[landmark_id]
            z_meas[i] = z[0]
            for j in range(2 * n + 1):
                dx = sigma_points[j, 0] - lx
                dy = sigma_points[j, 1] - ly
                z_sigma[j, i] = np.hypot(dx, dy)

        # обновление
        z_mean = np.dot(W, z_sigma)      # взвешенное среднее по расстоянию (в лекции u)
        z_diff = z_sigma - z_mean        # в лекции: y - u
        x_diff = sigma_points - state    # в лекции: X - m

        R = np.eye(num_meas) * sigma
        C = (x_diff.T * W) @ z_diff
        S = (z_diff.T * W) @ z_diff + R  # ковариация измерений
        K = C @ np.linalg.inv(S)         # калмановский коэффициент

        state += K @ (z_meas - z_mean)
        P -= K @ S @ K.T

    return state, P


def main():
    # Инициализация
    landmarks, sensor_data = load_data()

    m = np.array([0.0, 0.0, 0.0])  # [x, y, theta] - вектор средних
    P = np.eye(3) * 0.01           # Матрица ковариации - разброс от m

    # Обработка данных
    ekf_m, ukf_m = m.copy(), m.copy()
    ekf_P, ukf_P = P.copy(), P.copy()

    ekf_states, ukf_states = [], []
    for odometry, measurements in sensor_data.items():
        ekf_m, ekf_P = ekf(ekf_m, ekf_P, odometry, landmarks, measurements)
        ukf_m, ukf_P = ukf(ukf_m, ukf_P, odometry, landmarks, measurements)

        ekf_states.append(ekf_m.copy())
        ukf_states.append(ukf_m.copy())

    # Визуализация
    ekf_states = np.array(ekf_states)
    ukf_states = np.array(ukf_states)

    draw(ekf_states, ukf_states, landmarks)


if __name__ == "__main__":
    main()
