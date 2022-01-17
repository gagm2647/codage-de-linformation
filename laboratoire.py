import math
import random

from scipy.io import wavfile
import scipy.signal as sc
import numpy as np
import matplotlib.pyplot as plt
import librosa

import warnings

warnings.filterwarnings("ignore")


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


def write_wav_file(data, wfile: str, fs: int):
    if wfile.endswith('.wav'):
        wavfile.write(wfile, fs, data)
    else:
        wavfile.write(wfile + '.wav', fs, data)


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
    for k in range(ech_len - fc2, ech_len - fc1):
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
    for i in range(0, window_count):
        y = []

        fenetre_hanning(data_in=signal, data_out=y, curr_window=i, window_size=window_len, hanning=window)

        # FFT
        fft_y = np.fft.fft(y)
        # plot_fft(fft_y)
        out.append(fft_y.copy())


def overlap_low_pass(signal, filtered_signal, window_count, window_len):
    half_window_len = window_len // 2
    for i in range(0, window_count):
        y = signal[i].copy()
        filtre_passe_bas_parfait(y, math.floor(half_window_len / 2), math.floor(3 * half_window_len / 2))
        # plot_fft(y, "Window : " + str(i) + " - low-pass")
        filtered_signal.append(y)


def overlap_high_pass(signal, filtered_signal, window_count, window_len):
    half_window_len = window_len // 2

    for i in range(0, window_count):
        y = signal[i].copy()
        filtre_passe_haut_parfait(y, math.floor(half_window_len / 2), math.floor(3 * half_window_len / 2),
                                  window_len)
        # plot_fft(y, "Window : " + str(i) + " - high-pass")
        filtered_signal.append(y)


def overlap_cut_band(signal, filtered_signal, window_count, window_len, fmin, fmax):
    for i in range(0, window_count):
        y = signal[i].copy()
        filtre_coupe_bande_parfait(y, fmin, fmax)

        # plot_fft(y, "Window : " + str(i) + " - cut-band 300Hz to 3400Hz")
        filtered_signal.append(y)


def add_overlapping_window(signal1, signal2, out, window_len):
    half_window_len = window_len // 2
    for i in range(0, window_len):
        if i > half_window_len:
            out.append(signal1[i] + signal2[i - half_window_len])
        else:
            out.append(signal1[i])


def num_2():
    fs, data = open_wav_file('sound_files\\yellow_48k.wav')
    data = data / max(data)
    data = data[:len(data)//4]
    plt.figure()
    plt.title("Signal original")
    plt.plot(data)
    window_len = 1024
    half_window = round(window_len / 2)

    # fenetre Hanning
    hanning = np.hanning(window_len)

    # apply window
    nb_window = round(2 * len(data) / window_len) - 1

    windowed_filtered_signal = []

    overlap_filtering(data, windowed_filtered_signal, nb_window, window_len, hanning)

    # filtrage bas
    windowed_filtered_signal_low_pass = []
    overlap_low_pass(windowed_filtered_signal, windowed_filtered_signal_low_pass, nb_window, window_len)

    # filtrage haut
    windowed_filtered_signal_high_pass = []
    overlap_high_pass(windowed_filtered_signal, windowed_filtered_signal_high_pass, nb_window, window_len)

    # filtrage coupe bande
    windowed_filtered_signal_cut_band = []
    fmin = round(300 * half_window / fs)
    fmax = round(3400 * half_window / fs)
    overlap_cut_band(windowed_filtered_signal, windowed_filtered_signal_cut_band, nb_window, window_len, fmin, fmax)

    #numéro 3
    num_3(windowed_filtered_signal, fs, window_len)

    # IFFT
    signal_low_pass = np.fft.ifft(windowed_filtered_signal_low_pass)
    signal_high_pass = np.fft.ifft(windowed_filtered_signal_high_pass)
    signal_cut_band = np.fft.ifft(windowed_filtered_signal_cut_band)

    # Add overlapping
    signal_low_pass_added = []
    signal_high_pass_added = []
    signal_cut_band_added = []

    for i in range(0, nb_window - 1):
        add_overlapping_window(signal_low_pass[i], signal_low_pass[i + 1], signal_low_pass_added, window_len)
        add_overlapping_window(signal_high_pass[i], signal_high_pass[i + 1], signal_high_pass_added, window_len)
        add_overlapping_window(signal_cut_band[i], signal_cut_band[i + 1], signal_cut_band_added, window_len)

    # Signal filtered
    plt.figure()
    plt.title("Signal filtered by low-pass")
    plt.plot(signal_low_pass_added)

    plt.figure()
    plt.title("Signal filtered by high-pass")
    plt.plot(signal_high_pass_added)

    plt.figure()
    plt.title("Signal filtered by cut-band")
    plt.plot(signal_cut_band_added)

    # Data convertion for wavfile
    lpa = np.abs(np.array(signal_low_pass_added)).astype(np.float32)
    hpa = np.abs(np.array(signal_high_pass_added)).astype(np.float32)
    cba = np.abs(np.array(signal_cut_band_added)).astype(np.float32)

    # Output sound file
    write_wav_file(lpa, 'signal_low_pass_added.wav', fs)
    write_wav_file(hpa, 'signal_high_pass_added.wav', fs)
    write_wav_file(cba, 'signal_cut_band_added.wav', fs)
    plt.show()

    return 1


def lpc_window(signal, order: int):
    a = librosa.lpc(signal, order)
    b = np.hstack([[0], -1 * a[1:]])
    y_hat = sc.lfilter(b, [1], signal)

    fig, ax = plt.subplots()
    ax.plot(signal)
    ax.plot(y_hat, linestyle='--')
    ax.legend(['y', 'y_hat'])
    ax.set_title('LP Model Forward Prediction')
    return y_hat.copy()


def return_ech_number(time_in_ms: int, fs: int, N: int):
    f = 1000 / time_in_ms
    m = round(f * N / fs)
    return m

def num_3(signal_avec_fenetre, fs, N):
    m = 1  # filter order

    predicted_signal = []
    for i in range(0, len(signal_avec_fenetre)):
        signal_abs = np.abs(signal_avec_fenetre[i])
        predicted_signal.append(lpc_window(signal_abs, m))

    return 2


def num_4():
    return 3
