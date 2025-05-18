import numpy as np
from scipy import signal, optimize

# Параметры системы
c = 1125                    # скорость звука (фут/с)
fs = 100000                 # частота дискретизации (Гц)
room_size = (20, 20, 10)    # размеры комнаты (x, y, z)


def load_data():
    s_records = []
    with open("../data_files/Transmitter.txt") as f:
        for line in f:
            if line.strip():  # пропускаем пустые строки
                s_records.append([float(x) for x in line.split()])
    s_records = np.array(s_records, dtype=np.float64)

    with open("../data_files/Receiver.txt") as f:
        received = np.array(f.read().strip().split(), dtype=np.float64)

    return s_records, received


def calculate_distances(s, r):
    distances = []
    for i in range(s.shape[0]):
        corr = signal.correlate(r, s[i])                    # вычисление взаимной корреляции
        lags = signal.correlation_lags(len(r), len(s[i]))   # все возможные сдвиги
        T = lags[np.argmax(corr)] / fs                      # сдвиг с наибольшей корреляцией
        R = T * c
        print(f'Задержка до {i+1} громкоговорителя: {T:.5f} сек. Расстояние: {R:.2f} футов')
        distances.append(R)
    return np.array(distances)


def find_position(pos, speakers_pos, distances):
    dst_pos = np.zeros(4)
    for i in range(4):
        dst_pos[i] = np.linalg.norm(speakers_pos[i] - pos)  # евклидово расстояние
    return dst_pos - distances


def main():
    speakers_pos = np.array([
        [0, 0, room_size[2]],                        # громкоговоритель 1
        [room_size[0], 0, room_size[2]],             # громкоговоритель 2
        [room_size[0], room_size[1], room_size[2]],  # громкоговоритель 3
        [0, room_size[1], room_size[2]]              # громкоговоритель 4
    ])

    s, r = load_data()
    distances = calculate_distances(s, r)

    start_pos = np.mean(speakers_pos, 0)
    bounds = ([0, 0, 0], [room_size[0], room_size[1], room_size[2]])
    result = optimize.least_squares(
        fun=find_position,
        x0=start_pos,
        args=(speakers_pos, distances),
        bounds=bounds,
    )
    print(f"Найденные координаты объекта (x, y, z): {result.x}")


if __name__ == "__main__":
    main()
