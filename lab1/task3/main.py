import numpy as np
from scipy.stats import multivariate_normal

# Параметры модели
mu0 = 4 * np.pi * 1e-7
L = 9
h = 0.1
lambda_p = 0.01
delta_q = 0.01
sigma_v = 1e-7
Gamma = sigma_v ** 2 * np.eye(L)
N_particles = 1000

# Генерация сенсоров
x_sensors = np.linspace(-0.1, 0.1, 3)
y_sensors = np.linspace(-0.1, 0.1, 3)
r_j = np.array([[x, y, h] for x in x_sensors for y in y_sensors])


def compute_G(p):
    delta = r_j[:, :2] - p
    norms = np.linalg.norm(np.hstack([delta, h * np.ones((L, 1))]), axis=1) ** 3
    return mu0 / (4 * np.pi) * np.column_stack([delta[:, 1] / norms, -delta[:, 0] / norms])


# Генерация истинной траектории
T = 100
true_p = np.cumsum(np.random.normal(0, lambda_p, (T, 2)), axis=0)
true_q = np.cumsum(np.random.normal(0, delta_q, (T, 2)), axis=0)

# Генерация наблюдений
y = np.array([compute_G(p) @ q + np.random.multivariate_normal(np.zeros(L), Gamma)
              for p, q in zip(true_p, true_q)])


# Реализация RBPF
class Particle:
    def __init__(self, p, q_mean, q_cov):
        self.p = p
        self.q_mean = q_mean
        self.q_cov = q_cov


particles = [Particle(true_p[0] + np.random.normal(0, 0.01, 2),
                      true_q[0] + np.random.normal(0, 0.01, 2),
                      np.eye(2) * 0.01 ** 2) for _ in range(N_particles)]

estimated_p = np.zeros((T, 2))
estimated_q = np.zeros((T, 2))

for k in range(T):
    # Prediction step
    for p in particles:
        p.p += np.random.normal(0, lambda_p, 2)
        p.q_cov += np.eye(2) * delta_q ** 2

    # Update step
    log_weights = []
    for p in particles:
        G = compute_G(p.p)
        S = G @ p.q_cov @ G.T + Gamma
        try:
            K = p.q_cov @ G.T @ np.linalg.inv(S)
            p.q_mean += K @ (y[k] - G @ p.q_mean)
            p.q_cov -= K @ G @ p.q_cov
            log_weights.append(multivariate_normal.logpdf(y[k], G @ p.q_mean, S))
        except:
            log_weights.append(-np.inf)

    # Resampling
    weights = np.exp(log_weights - np.max(log_weights))
    weights /= weights.sum()
    indices = np.random.choice(N_particles, size=N_particles, p=weights)
    particles = [Particle(particles[i].p.copy(),
                          particles[i].q_mean.copy(),
                          particles[i].q_cov.copy()) for i in indices]

    # State estimation
    estimated_p[k] = np.mean([p.p for p in particles], axis=0)
    estimated_q[k] = np.mean([p.q_mean for p in particles], axis=0)

# Визуализация
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.subplot(211)
plt.plot(true_p[:, 0], true_p[:, 1], label='Истинная траектория')
plt.plot(estimated_p[:, 0], estimated_p[:, 1], '--', label='Оценка')
plt.legend()

plt.subplot(212)
plt.plot(true_q, label=['q1 истинный', 'q2 истинный'])
plt.plot(estimated_q, '--', label=['q1 оценка', 'q2 оценка'])
plt.legend()
plt.show()
