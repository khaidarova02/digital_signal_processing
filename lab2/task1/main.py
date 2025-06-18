import numpy as np
import matplotlib.pyplot as plt


def draw(original_time, original_signal, samples_time, discrete_signal, reconstructed_signal):
    fig, axes = plt.subplots(1, 3)

    axes[0].plot(original_time, original_signal)
    axes[0].set_ylim(-2, 2)
    axes[0].set_title("Оригинальный сигнал")

    axes[1].plot(samples_time, discrete_signal)
    axes[1].set_ylim(-2, 2)
    axes[1].set_title("Дискретный сигнал")

    axes[2].plot(original_time, reconstructed_signal)
    axes[2].set_ylim(-2, 2)
    axes[2].set_title("Восстановленный сигнал")

    plt.tight_layout()
    plt.show()


def main():
    # Оригинальный сигнал
    fmax = 2                  # макс. частота дискретизации (Гц)
    w0 = 2 * np.pi * fmax     # угловая частота
    original_time = np.arange(0, 1, 0.001)
    original_signal = np.sin(original_time * w0)  # = x(t)

    # Дискретный сигнал
    fs = 2 * fmax + 1      # дискретная частота
    T = 1 / fs             # T < 1 / (2 * fmax) - условие теоремы отчетов
    samples_time = np.arange(0, 1, T)
    discrete_signal = np.sin(w0 * samples_time)

    # Восстановленный сигнал
    reconstructed_signal = 0
    for i in samples_time:
        # np.sinc = sin(pi * x) / (pi * x)
        # (pi / T)(t - kT) -> (t - kT) / T = fs(t - kT)
        sinc_values = np.sinc(fs * (original_time - i))
        reconstructed_signal += np.sin(i * w0) * sinc_values

    # Визуализация
    draw(original_time, original_signal, samples_time, discrete_signal, reconstructed_signal)


if __name__ == "__main__":
    main()
