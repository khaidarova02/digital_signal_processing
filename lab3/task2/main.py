import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.signal import firwin, lfilter


def draw_tune(frequency, amplitude, filtered_spectrum):
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


def draw_ecg(time, signal, freq, amp, filtered, filtered_signal):
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
    sampling, signal = wav.read('../data_files/tune.wav')   # частота дискретизации, массив отсчетов сигнала
    n = signal.shape[0]                                     # количество отсчетов

    # Фильтрация сигнала
    # numtaps — количество коэффициентов фильтра (порядок фильтра + 1)
    # cutoff - нормированная частота среза (9000 / (fs / 2) ~ 0.4)
    coef_filter = firwin(numtaps=43, cutoff=0.04, window='blackman', pass_zero='lowpass')
    filtered_signal = lfilter(coef_filter, 1.0, signal).astype(np.int16)  # a - знаменатель передаточной функции

    frequency = np.fft.fftfreq(n, 1 / sampling)      # вычисления частот (кол-во отсчетов + шаг дискретизации)
    amplitude_filter = np.abs(np.fft.fft(filtered_signal))  # амплитудный спектр
    amplitude = np.abs(np.fft.fft(signal))

    # Запись восстановленного сигнала и визуализация
    wav.write('../data_files/tune_filtered.wav', sampling, filtered_signal)
    draw_tune(frequency, amplitude, amplitude_filter)

    # Считываем данные
    time, signal = [], []
    with open("../data_files/ecg.dat") as file:
        for line in file.readlines():
            vals = line.split()
            time.append(float(vals[0]))
            signal.append(float(vals[1]))

    n = len(signal)  # количество отсчетов
    T = time[2] - time[1]
    spectrum = np.fft.fft(signal)
    amplitude = np.abs(spectrum)
    frequency = np.fft.fftfreq(n, T)

    # Фильтрация сигнала
    # numtaps — количество коэффициентов фильтра (порядок фильтра + 1)
    # cutoff - нормированная частота среза
    coef_filter = firwin(numtaps=87, cutoff=[0.06, 0.15], window='blackman', pass_zero='bandstop')
    filtered_signal = lfilter(coef_filter, 1.0, signal)  # a - знаменатель передаточной функции
    filtered = np.real(np.fft.fft(filtered_signal))

    # Визуализация
    draw_ecg(time, signal, frequency, amplitude, filtered, filtered_signal)


if __name__ == "__main__":
    main()
