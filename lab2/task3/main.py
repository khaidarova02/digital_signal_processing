import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav


def draw(frequency, amplitude, filtered_spectrum):
    fig, axes = plt.subplots(1, 2)

    axes[0].plot(frequency, amplitude)
    axes[0].set_title("tune.wav")
    axes[0].set_ylabel('Амплитуда')
    axes[0].set_xlabel('Частота')

    axes[1].plot(frequency, np.abs(filtered_spectrum))
    axes[1].set_title("tune_filtered.wav")
    axes[1].set_ylabel('Амплитуда')
    axes[1].set_xlabel('Частота')
    plt.show()


def main():
    sampling, signal = wav.read('../data_files/tune.wav')   # частота дискретизации, массив отсчетов сигнала
    n = signal.shape[0]                                     # количество отсчетов

    # Оригинальный сигнал
    spectrum = np.fft.fft(signal)  # прямое БПФ
    amplitude = np.abs(spectrum)   # амплитудный спектр
    frequency = np.fft.fftfreq(n, 1 / sampling)  # вычисления частот (кол-во отсчетов + шаг дискретизации)

    # Восстановленный сигнал
    frequency_mask = np.array([0 if abs(f) > 9000 else 1 for f in frequency])   # маска частот
    filtered_spectrum = np.array(spectrum) * frequency_mask                     # фильтрация высоких частот
    filtered_signal = np.real(np.fft.ifft(filtered_spectrum)).astype(np.int16)  # обратное БПФ

    # Запись восстановленного сигнала и визуализация
    wav.write('../data_files/tune_filtered.wav', sampling, filtered_signal)
    draw(frequency, amplitude, filtered_spectrum)


if __name__ == "__main__":
    main()
