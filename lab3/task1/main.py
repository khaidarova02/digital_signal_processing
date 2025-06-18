import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal


fs = 8000  # частота дискретизации 8 кГц
N = 501    # порядок фильтра (нечетный для линейной ФЧХ)
L = 2      # длительность сигнала в секундах для анализа


def draw(freq, desired, lib, custom):
    """Визуализация результатов"""
    plt.figure(figsize=(10, 6))
    plt.xlim(0, 2000)  # Ограничиваем диапазон частот до 2000 Гц
    plt.plot(freq, np.abs(desired), label="Идеальная АЧХ", linewidth=2)
    plt.plot(freq, np.abs(lib), label="Библиотечная (signal.firls)", linestyle='--')
    plt.plot(freq, np.abs(custom), label="МНК", linestyle=':')
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Амплитуда')
    plt.title("Сравнение АЧХ фильтров")
    plt.grid(True)
    plt.legend()
    plt.show()


def H_f(f):
    H = np.zeros_like(f)
    for i, freq in enumerate(f):
        if 100 < freq <= 300:
            H[i] = 2.0
        elif 300 < freq <= 700:
            H[i] = -freq / 200 + 7 / 2
        elif 700 < freq <= 1200:
            H[i] = freq / 500 - 7 / 5
        elif 1200 < freq <= 1500:
            H[i] = 1.0
    return H


def bandpass_filter(freq):
    """Реализация КИХ-фильтра методом наименьших квадратов"""
    w = freq / fs * 2 * np.pi  # перевод в радианы

    # Желаемая АЧХ
    a = H_f(freq).T

    # Матрица проекции F = 2 * sin(w * (M-n)) - 3 тип (N – нечетное, h[n] = −h[N − 1 − n])
    M = (N - 1) // 2
    F = np.zeros((N // 2, M))
    for i in range(M):
        F[:, i] = 2 * np.sin(w * (i + 1))

    # Решение методом наименьших квадратов h = (F.T * F)^(-1) * F.T * a
    h_half = np.linalg.inv(F.T @ F) @ F.T @ a
    h = np.hstack((h_half[::-1], [0], -h_half))

    return h


def main():
    # Расчет КИХ-фильтра с помощью библиотечной функции
    # bands = np.array([0, 100, 101, 300, 301, 700, 701, 1200, 1201, 1500, 1501, fs // 2])
    bands = np.linspace(0, fs / 2, N // 2)
    desired = H_f(bands)
    lib_filter = signal.firls(N, bands, desired, fs=fs)

    # Расчет КИХ-фильтра с помощью МНК
    freq = np.linspace(0, fs / 2, N // 2)   # дискретные частоты в диапазоне [0, fs/2]
    custom_filter = bandpass_filter(freq)

    # Расчет частотных характеристик
    _, lib_response = signal.freqz(lib_filter, 1, worN=freq, fs=fs)
    _, custom_response = signal.freqz(custom_filter, 1, worN=freq, fs=fs)
    ideal_response = H_f(freq)

    # Визуализация
    draw(freq, ideal_response, lib_response, custom_response)


if __name__ == "__main__":
    main()
