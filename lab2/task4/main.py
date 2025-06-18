import numpy as np
import matplotlib.pyplot as plt


def draw(time, signal, freq, amp, filtered, filtered_signal):
    fig, axes = plt.subplots(2, 2)

    axes[0, 0].set_title("Оригинальный ЭКГ")
    axes[0, 0].plot(time[:4000], signal[:4000])

    axes[0, 1].plot(freq, amp)
    axes[0, 1].set_title("Оригинальный спектр")
    axes[0, 1].set_ylabel('Амплитуда')
    axes[0, 1].set_xlabel('Частота')
    axes[0, 1].set_xlim(-100, 100)

    axes[1, 1].plot(freq, np.abs(filtered))
    axes[1, 1].set_title("Отфильтрованный спектр")
    axes[1, 1].set_ylabel('Амплитуда')
    axes[1, 1].set_xlabel('Частота')
    axes[1, 1].set_xlim(-100, 100)

    axes[1, 0].set_title("Фильтрованный ЭКГ")
    axes[1, 0].plot(time[:4000], filtered_signal[:4000])
    plt.tight_layout()
    plt.show()


def main():
    # Считываем данные
    time, orig_signal = [], []
    with open("../data_files/ecg.dat") as file:
        for line in file.readlines():
            vals = line.split()
            time.append(float(vals[0]))
            orig_signal.append(float(vals[1]))

    n = len(orig_signal)                # количество отсчетов
    T = time[2] - time[1]               # период
    spectrum = np.fft.fft(orig_signal)  # прямое БПФ
    amplitude = np.abs(spectrum)        # амплитудный спектр
    frequency = np.fft.fftfreq(n, T)    # частотная шкала

    # Фильтрация
    frequency_mask = np.array([0 if 49 < abs(f) < 51 else 1 for f in frequency])  # маска частот
    filtered = np.array(spectrum) * frequency_mask                                # фильтрация частот
    filtered_signal = np.real(np.fft.ifft(filtered))                              # обратное БПФ

    # Визуализация
    draw(time, orig_signal, frequency, amplitude, filtered, filtered_signal)


if __name__ == "__main__":
    main()
