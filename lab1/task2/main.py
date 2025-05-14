import numpy as np
import matplotlib.pyplot as plt


class EKF:
    def __init__(self, initial_state, initial_cov, Q, R):
        self.state = initial_state
        self.P = initial_cov
        self.Q = Q
        self.R = R

    def predict(self, delta_r1, delta_t, delta_r2):
        x, y, theta = self.state
        # Обновление состояния
        new_theta = theta + delta_r1 + delta_r2
        new_x = x + delta_t * np.cos(theta + delta_r1)
        new_y = y + delta_t * np.sin(theta + delta_r1)
        self.state = np.array([new_x, new_y, new_theta])

        # Якобиан F
        F = np.eye(3)
        F[0, 2] = -delta_t * np.sin(theta + delta_r1)
        F[1, 2] = delta_t * np.cos(theta + delta_r1)

        # Обновление ковариации
        self.P = F @ self.P @ F.T + self.Q
        self.state[2] = (self.state[2] + np.pi) % (2 * np.pi) - np.pi  # Нормализация угла

    def update(self, landmarks, measurements):
        for landmark_id, z_meas in measurements:
            lx, ly = landmarks[landmark_id]
            x, y, theta = self.state

            # Ожидаемое измерение
            z_pred = np.hypot(x - lx, y - ly)

            # Якобиан H
            H = np.zeros((1, 3))
            dx = x - lx
            dy = y - ly
            H[0, 0] = dx / z_pred
            H[0, 1] = dy / z_pred

            # Обновление Калмана
            S = H @ self.P @ H.T + self.R
            K = self.P @ H.T / S
            self.state += K.flatten() * (z_meas - z_pred)
            self.P = (np.eye(3) - K @ H) @ self.P


class UKF:
    def __init__(self, initial_state, initial_cov, Q, R, alpha=1e-3, beta=2, kappa=0):
        self.state = initial_state
        self.P = initial_cov
        self.Q = Q
        self.R = R
        self.n = 3
        self.lambda_ = alpha ** 2 * (self.n + kappa) - self.n
        self.weights = self._compute_weights(alpha, beta)

    def _compute_weights(self, alpha, beta):
        weights = np.zeros(2 * self.n + 1)
        weights[0] = self.lambda_ / (self.n + self.lambda_)
        weights[1:] = 1 / (2 * (self.n + self.lambda_))
        return weights

    def _compute_sigma_points(self):
        sigma_points = np.zeros((2 * self.n + 1, self.n))
        sigma_points[0] = self.state
        L = np.linalg.cholesky((self.n + self.lambda_) * self.P)
        for i in range(self.n):
            sigma_points[i + 1] = self.state + L[i]
            sigma_points[self.n + i + 1] = self.state - L[i]
        return sigma_points

    def predict(self, delta_r1, delta_t, delta_r2):
        sigma_points = self._compute_sigma_points()
        # Прогноз для каждой сигма-точки
        for i in range(2 * self.n + 1):
            x, y, theta = sigma_points[i]
            new_theta = theta + delta_r1 + delta_r2
            new_x = x + delta_t * np.cos(theta + delta_r1)
            new_y = y + delta_t * np.sin(theta + delta_r1)
            sigma_points[i] = [new_x, new_y, new_theta]

        # Обновление состояния и ковариации
        self.state = np.dot(self.weights, sigma_points)
        delta = sigma_points - self.state
        self.P = (delta.T * self.weights) @ delta + self.Q
        self.state[2] = (self.state[2] + np.pi) % (2 * np.pi) - np.pi

    def update(self, landmarks, measurements):
        for landmark_id, z_meas in measurements:
            sigma_points = self._compute_sigma_points()
            lx, ly = landmarks[landmark_id]

            # Преобразование сигма-точек в измерения
            z_sigma = np.array([np.hypot(pt[0] - lx, pt[1] - ly) for pt in sigma_points])

            # Вычисление статистик
            z_mean = np.dot(self.weights, z_sigma)
            z_diff = z_sigma - z_mean
            x_diff = sigma_points - self.state
            Pzz = (z_diff.T * self.weights) @ z_diff + self.R
            Pxz = (x_diff.T * self.weights) @ z_diff

            # Обновление Калмана
            K = Pxz / Pzz
            self.state += K * (z_meas - z_mean)
            self.P -= K * Pzz * K.T


def load_data(landmarks_path, sensor_path):
    landmarks = {}
    with open(landmarks_path) as f:
        for line in f:
            id_, x, y = map(float, line.strip().split())
            landmarks[int(id_)] = (x, y)

    sensor_data = []
    with open(sensor_path) as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            t = parts[0]
            controls = parts[1:4]
            measurements = [(int(parts[i]), parts[i + 1])
                            for i in range(4, len(parts), 2)]
            sensor_data.append((controls, measurements))
    return landmarks, sensor_data


# Инициализация
landmarks, sensor_data = load_data('data_files/landmarks.dat', 'data_files/sensor_data_ekf.dat')
initial_state = np.array([0.0, 0.0, 0.0])
initial_cov = np.eye(3) * 1e-3
Q = np.eye(3) * 0.2
R = 0.2

# Создание фильтров
ekf = EKF(initial_state.copy(), initial_cov.copy(), Q, R)
ukf = UKF(initial_state.copy(), initial_cov.copy(), Q, R)

# Обработка данных
ekf_states, ukf_states = [], []
for controls, measurements in sensor_data:
    delta_r1, delta_t, delta_r2 = controls

    # Предсказание
    ekf.predict(delta_r1, delta_t, delta_r2)
    ukf.predict(delta_r1, delta_t, delta_r2)

    # Коррекция
    if measurements:
        ekf.update(landmarks, measurements)
        ukf.update(landmarks, measurements)

    ekf_states.append(ekf.state.copy())
    ukf_states.append(ukf.state.copy())

# Визуализация
ekf_states = np.array(ekf_states)
ukf_states = np.array(ukf_states)

plt.figure(figsize=(12, 8))
plt.plot(ekf_states[:, 0], ekf_states[:, 1], label='EKF')
plt.plot(ukf_states[:, 0], ukf_states[:, 1], '--', label='UKF')
plt.scatter(*zip(*landmarks.values()), marker='*', c='red', label='Landmarks')
plt.legend()
plt.title('Сравнение траекторий EKF и UKF')
plt.xlabel('X координата')
plt.ylabel('Y координата')
plt.grid(True)
plt.show()