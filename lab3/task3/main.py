import numpy as np
from scipy.signal import firwin2, freqz
import matplotlib.pyplot as plt

# Параметры
Fs = 44000  # Частота дискретизации
Fc = 30000  # Частота среза микрофона


def draw(h, w_filt, H_filt, w, H):
    plt.subplot(1, 2, 1)
    plt.plot(h)
    plt.title('Импульсная характеристика фильтра')
    plt.xlabel('Номер отсчета (n)')
    plt.ylabel('Амплитуда h[n]')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(w_filt, np.abs(H_filt), label='Полученная ЧХ')
    plt.plot(w, H, label='Желаемая ЧХ')
    plt.title('Частотная характеристика')
    plt.xlabel('Цифровая частота (рад/отсчет)')
    plt.ylabel('Амплитуда |H(ω)|')
    plt.legend()
    plt.grid(True)

    plt.show()


def main():
    # Желаемая АЧХ
    w = np.linspace(0, np.pi, 1000)
    H = 1 / (1 - (Fs * w) / (Fc * 2 * np.pi))  # компенсирующая ЧХ (обратная к характеристике микрофона Ha(F))

    # Расчет КИХ-фильтра (по умол. Хэмминг)
    # автоматически обрезается частоты выше Найквиста (Fs/2 = 22 кГц).
    h = firwin2(numtaps=101, freq=w / np.pi, gain=H)

    # Расчет массива нормированных частот и КЧХ
    w_filt, H_filt = freqz(h)

    # Визуализация
    draw(h, w_filt, H_filt, w, H)


if __name__ == "__main__":
    main()
