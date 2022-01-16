import math
import random

from scipy.io import wavfile
import scipy as sc
import numpy as np
import matplotlib.pyplot as plt


# Débuzz louis
def open_wav_file(wfile: str):
    """
    Inputs
        filename (str) : name of the .wav file to open
    Returns
        rate (int) : sample rate of WAV File
        data (numpy array) : Data read from .wav file
        """
    if wfile.endswith('.wav'):
        fs, data = wavfile.read(wfile)
        return fs, data
    else:
        print('Wrong file type. Type accepted : .wav')
        return None, None


def autocorrelation(nbTermes: int, data):
    _autocorrelation = np.empty(nbTermes)
    for i in range(nbTermes):
        for j in range(len(data)):
            _autocorrelation[i] = data[j] * data[j - i] + _autocorrelation[i]
    return _autocorrelation


def plot_data_cloud(data):
    plt.figure()
    plt.plot(data[:-1], data[1:], 'ro')


def plot_data_cloud_3d(data):
    plt.figure()
    plt.plot(data, 'ro')


def plot_autocorrelation(data):
    lags, r, line, b = plt.acorr(data)
    plt.figure()
    plt.plot(lags, r)
    return 0


def num_1():
    fs, data = open_wav_file('sound_files\\x_test.wav')
    data = data / max(data)

    ret_val = plt.xcorr(data.astype('float'), data.astype('float'))
    _autocorrelation = ret_val[1]

    mid = math.floor(len(_autocorrelation) / 2)
    print("R1/R0 : ")
    print(_autocorrelation[mid + 1] / _autocorrelation[mid])
    plot_data_cloud(data)

    C_data = [[_autocorrelation[mid], _autocorrelation[mid + 1]],
              [_autocorrelation[mid + 1], _autocorrelation[mid]]]  # prendre le milieu plus celui apres

    valeurs_propres, vecteurs_propres = np.linalg.eig(C_data)
    print(vecteurs_propres)
    y = []

    for i, d in enumerate(data[:-1]):
        pair = [data[i], data[i + 1]]
        y.append(np.dot(vecteurs_propres, pair))

    plot_data_cloud_3d(y)
    print("On peut voir que les pairs y[n] y[n+1] montrent que le signal est décorrellé.")
    plt.show()
    return 0


def filtre_passe_bas_parfait(data, fb: int, fh: int):
    for i in range(fb, fh):
        data[i] = 0


def filtre_passe_haut_parfait(data, fh1: int, fh2: int, fmax: int):
    for k in range(0, fh1):
        data[k] = 0
    for k in range(fh2, fmax):
        data[k] = 0


def filtre_coupe_bande_parfait(data, fc1: int, fc2: int):
    ech_len = len(data)
    for k in range(fc1, fc2):
        data[k] = 0
    for k in range( ech_len - fc2, ech_len - fc1):
        data[k] = 0


def plot_pb_ph_stem(ph, pb):
    plt.figure()
    plt.title("Signal fenetre passe-bas")
    passe_bas = plt.stem(pb, linefmt='--m', markerfmt='om')
    passe_haut = plt.stem(ph, linefmt='--b', markerfmt='ob')
    plt.legend([passe_bas, passe_haut], ['Passe-bas', 'Passe-haut'])


def plot_fft(data, title=""):
    plt.figure()
    plt.title("FFT signal : " + title)
    plt.stem(data)


def fenetre_hanning(data_in, data_out, curr_window: int, window_size: int, hanning=None):
    if hanning is None:
        hanning = np.hanning(window_size)

    half_window = window_size // 2
    for i in range(0, window_size):
        data_out.append(hanning[i] * data_in[curr_window * half_window + i])


def overlap_filtering(signal, out, window_count, window_len, window):
    for i in range(0, min(window_count, 3)):
        y = []

        fenetre_hanning(data_in=signal, data_out=y, curr_window=i, window_size=window_len, hanning=window)

        # FFT
        fft_y = np.fft.fft(y)
        plot_fft(fft_y)
        out.append(fft_y.copy())


def overlap_low_pass(signal, filtered_signal, window_count, window_len):
    half_window_len = window_len // 2
    for i in range(0, window_count):
        filtered_signal = signal[i].copy()
        filtre_passe_bas_parfait(filtered_signal, math.floor(half_window_len / 2), math.floor(3 * half_window_len / 2))
        plot_fft(filtered_signal, "Window : " + str(i) + " - low-pass")


def overlap_high_pass(signal, filtered_signal, window_count, window_len):
    half_window_len = window_len // 2

    for i in range(0, window_count):
        filtered_signal = signal[i].copy()
        filtre_passe_haut_parfait(filtered_signal, math.floor(half_window_len / 2), math.floor(3 * half_window_len / 2),
                                  window_len)
        plot_fft(filtered_signal, "Window : " + str(i) + " - high-pass")


def overlap_cut_band(signal, filtered_signal, window_count, window_len, fmin, fmax):
    half_window_len = window_len // 2

    for i in range(0, window_count):
        filtered_signal = signal[i].copy()
        filtre_coupe_bande_parfait(filtered_signal, fmin, fmax)

        plot_fft(filtered_signal, "Window : " + str(i) + " - cut-band 300Hz to 3400Hz")


def num_2():
    fs, data = open_wav_file('sound_files\\yellow_48k.wav')
    data = data / max(data)
    window_len = 1024
    half_window = round(window_len / 2)

    # fenetre Hanning
    hanning = np.hanning(window_len)

    # apply window
    nb_window = round(2 * len(data) / window_len)

    windowed_filtered_signal = []

    overlap_filtering(data, windowed_filtered_signal, nb_window, window_len, hanning)

    # filtrage bas
    windowed_filtered_signal_low_pass = []
    overlap_low_pass(windowed_filtered_signal, windowed_filtered_signal_low_pass, min(nb_window, 3), window_len)

    # filtrage haut
    windowed_filtered_signal_high_pass = []
    overlap_high_pass(windowed_filtered_signal, windowed_filtered_signal_high_pass, min(nb_window, 3), window_len)

    # filtrage coupe bande
    windowed_filtered_signal_cut_band = []
    fmin = round(300 * half_window / fs)
    fmax = round(3400 * half_window / fs)
    overlap_cut_band(windowed_filtered_signal, windowed_filtered_signal_cut_band, min(nb_window, 3), window_len, fmin, fmax)

    # IFFT
    plt.show()
    return 1


def num_3():
    return 2


def num_4():
    return 3
