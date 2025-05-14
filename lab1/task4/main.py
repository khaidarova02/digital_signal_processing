import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R


def load_data(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def skew_symmetric(a):
    return np.array([[0, -a[2], a[1]],
                     [a[2], 0, -a[0]],
                     [-a[1], a[0], 0]])


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])


def q_from_theta(theta):
    angle = np.linalg.norm(theta)
    if angle < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = theta / angle
    return np.concatenate([[np.cos(angle / 2)], axis * np.sin(angle / 2)])


def rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x ** 2 - 2 * y ** 2]
    ])


class ESKF:
    def __init__(self, p0, v0, q0, dt, sigma_acc, sigma_gyro, sigma_gnss, sigma_lidar):
        self.dt = dt
        self.g = np.array([0, 0, -9.81])

        # Номинальное состояние
        self.p = p0.copy()
        self.v = v0.copy()
        self.q = q0.copy()

        # Матрицы шумов
        self.Q = np.diag([sigma_acc ** 2] * 3 + [sigma_gyro ** 2] * 3) * dt ** 2
        self.R_gnss = np.eye(3) * sigma_gnss ** 2
        self.R_lidar = np.eye(3) * sigma_lidar ** 2

        # Ковариационная матрица ошибок
        self.P = np.eye(9) * 0.1

        # Матрица преобразования лидара
        self.C = np.array([[0.99376, -0.09722, 0.05466],
                           [0.09971, 0.99401, -0.04475],
                           [-0.04998, 0.04992, 0.9975]])
        self.t = np.array([0.5, 0.1, 0.5])

    def predict(self, f, omega):
        # Обновление номинального состояния
        R_q = rotation_matrix(self.q)
        acc_global = R_q @ f + self.g

        self.p += self.v * self.dt + 0.5 * acc_global * self.dt ** 2
        self.v += acc_global * self.dt
        delta_q = q_from_theta(omega * self.dt)
        self.q = quat_mult(self.q, delta_q)
        self.q /= np.linalg.norm(self.q)

        # Матрица F
        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * self.dt
        F[3:6, 6:9] = -skew_symmetric(R_q @ f) * self.dt

        # Матрица L
        L = np.zeros((9, 6))
        L[3:6, 0:3] = np.eye(3)
        L[6:9, 3:6] = np.eye(3)

        # Обновление ковариации
        self.P = F @ self.P @ F.T + L @ self.Q @ L.T

    def update(self, z, H, R):
        # Обновление ошибок
        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + R)
        delta_x = K @ (z - self.p)

        self.P = (np.eye(9) - K @ H) @ self.P

        # Инжекция ошибок в номинальное состояние
        self.p += delta_x[0:3]
        self.v += delta_x[3:6]
        delta_theta = delta_x[6:9]
        self.q = quat_mult(q_from_theta(delta_theta), self.q)
        self.q /= np.linalg.norm(self.q)

        # Сброс ошибок
        self.P = (np.eye(9) - K @ H) @ self.P @ (np.eye(9) - K @ H).T + K @ R @ K.T

    def process_measurement(self, f, omega, z_gnss=None, z_lidar=None):
        self.predict(f, omega)

        if z_gnss is not None:
            H = np.hstack([np.eye(3), np.zeros((3, 6))])
            self.update(z_gnss, H, self.R_gnss)

        if z_lidar is not None:
            z_lidar_corrected = self.C @ z_lidar + self.t
            H = np.hstack([np.eye(3), np.zeros((3, 6))])
            self.update(z_lidar_corrected, H, self.R_lidar)

        return self.p.copy(), self.v.copy(), self.q.copy()


# Пример использования
data = load_data('data_files/data/data.pkl')
dt = data['dt']
sigma_acc = data['sigma_acc']
sigma_gyro = data['sigma_gyro']
sigma_gnss = data['sigma_gnss']
sigma_lidar = data['sigma_lidar']

# Инициализация из Ground Truth
p0 = data['gt_p'][0]
v0 = data['gt_v'][0]
q0 = data['gt_q'][0]

eskf = ESKF(p0, v0, q0, dt, sigma_acc, sigma_gyro, sigma_gnss, sigma_lidar)

estimated_traj = []
for i in range(1, len(data['imu_time'])):
    f = data['imu_acc'][i - 1]
    omega = data['imu_gyro'][i - 1]

    z_gnss = data['gnss_p'][i] if data['gnss_mask'][i] else None
    z_lidar = data['lidar_p'][i] if data['lidar_mask'][i] else None

    p_est, v_est, q_est = eskf.process_measurement(f, omega, z_gnss, z_lidar)
    estimated_traj.append(p_est)

# Оценка точности
ground_truth = np.array(data['gt_p'])
estimated_traj = np.array(estimated_traj)
error = np.linalg.norm(estimated_traj - ground_truth[1:], axis=1)
print(f"Средняя ошибка: {np.mean(error):.3f} м")
