import numpy as np
import scipy.io.wavfile as wav


def main():
    # Считываем данные
    sampling, signal = wav.read('../data_files/test5.wav')  # частота дискретизации, массив отсчетов сигнала
    spectrum = np.array(np.fft.fft(signal))  # прямое преобразование Фурье

    # Выделение спектров 4 равных файлов
    n = (len(spectrum) - 1) // 2  # центр симметрии
    n //= 4

    zero_val = spectrum[0]  # центр
    C = spectrum[1:n]
    B = spectrum[n:2*n]
    D = spectrum[2*n:3*n]
    A = spectrum[3*n:4*n]

    # Декодирование - получение исходного аудио
    decode = np.hstack((zero_val, A, B, C, D, np.conj(D)[::-1], np.conj(C)[::-1], np.conj(B)[::-1], np.conj(A)[::-1]))
    filtered_signal = np.real(np.fft.ifft(decode)).astype(np.int16)  # обратное преобразование Фурье

    wav.write('../data_files/test5_decode.wav', sampling, filtered_signal)


if __name__ == "__main__":
    main()
