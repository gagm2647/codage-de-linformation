import math
import random
from typing import List

from numpy import ndarray
from scipy.io import wavfile
import scipy.signal as sc
import numpy as np
import scipy.fft as scfft
import matplotlib.pyplot as plt
import librosa
import laboratoire as lab

import warnings

warnings.filterwarnings("ignore")


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


def trammeur_fenetreur(_signal, _frame_len, _hop_len):
    frames = librosa.util.frame(_signal, frame_length=_frame_len, hop_length=_hop_len)
    return (np.hanning(_frame_len).reshape(-1, 1) * frames).T


def reconstruction_signal(_windowed_frames, _hop_len):
    _signal_reconstruit = []
    for i, w in enumerate(_windowed_frames):
        if i == 0:
            for k in range(0, _hop_len):
                _signal_reconstruit.append(w[k])
        elif i + 1 < len(_windowed_frames):
            w2 = _windowed_frames[i + 1]
            for k in range(0, _hop_len):
                _signal_reconstruit.append(w[k + _hop_len] + w2[k])
        else:
            for k in range(0, _hop_len):
                _signal_reconstruit.append(w[k + _hop_len])
    return np.array(_signal_reconstruit)


def rehaussementDCT():
    # Analyse selon la technique
    # Extraction de paramètres (coefficient de transformée, coefficients de filtre prédicteurs, etc)
    # Modification des paramètres => Permet de retrouvée une ENVELOPPE SPECTRALE comprimée d'un facteur 2 à 3
    # Aucun changement sur la position des harmoniques du signal d'origine
    fs, raw = open_wav_file("sound_files/hel_fr2.wav")
    signal = soustraire_moyenne(normalisation_signal(raw))
    frame_len, hop_len = 882, 441
    windowed_frames = trammeur_fenetreur(signal, frame_len, hop_len)
    #s = reconstruction_signal(windowed_frames, hop_len)
    trames_traitees = []
    for i, trame in enumerate(windowed_frames):

        T = scfft.dct(trame)
        neg = T < 0
        Env = enveloppeSpectreDCT(abs(T), 45)
        E = T / Env
        Env2 = compressionSpectreDCT(Env, 3)

        trames_traitees.append(scfft.idct(Env2 * E))


    s = reconstruction_signal(trames_traitees, hop_len)

    #print(E)
    # plt.plot(s * max(raw))
    # plt.figure()
    # #plt.plot(E)
    # plt.plot(Env)
    # plt.plot(Env2)
    # plt.show()

    # signal_rehausse = overlappedDFT[range(0, len(overlappedDFT), 2)]
    write_wav_file(s, 'sound_files/dct.wav', fs)


    pass


def enveloppeSpectreDCT(amplitude_freqs: np.array, k: int = 10):
    a = 20 * np.log10(amplitude_freqs)
    Y = scfft.dct(a)
    indices = range(k, len(Y))
    Y[indices] = 0
    return 10 ** (scfft.idct(Y) / 20)


def compressionSpectreDCT(signal, steps=2):
    s = signal[range(0, len(signal), steps)]
    r = np.zeros(len(signal))
    r[range(0, len(s))] = s
    return r

def rehaussementDFT():
    # Analyse selon la technique
    # Extraction de paramètres (coefficient de transformée, coefficients de filtre prédicteurs, etc)
    # Modification des paramètres => Permet de retrouvée une ENVELOPPE SPECTRALE comprimée d'un facteur 2 à 3
    # Aucun changement sur la position des harmoniques du signal d'origine
    fs, raw = open_wav_file("sound_files/hel_fr2.wav")
    signal = soustraire_moyenne(normalisation_signal(raw))
    frame_len, hop_len = 882, 441
    windowed_frames = trammeur_fenetreur(signal, frame_len, hop_len)
    #s = reconstruction_signal(windowed_frames, hop_len)
    trames_traitees = []
    for i, trame in enumerate(windowed_frames):
        #b, a = sc.butter(3, 10000/fs, btype='low', output='ba')
        #trame = sc.filtfilt(b, a, trame)  # trame[range(hop_len, frame_len)] = 0
        #print(trame)
        T = np.fft.fft(trame)
        phase = np.angle(T)
        Env = enveloppeSpectrale(np.abs(T), 45)
       # print(len(T))
        E = np.abs(T) / Env
        # plt.figure()
        # plt.plot(E)
        # plt.show()
        Env2 = compressionSpectre(Env, 3)
        # plt.figure()
        # plt.plot(np.abs(T))


        # plt.show()
        #E2 = enveloppeSpectrale(np.abs(T2), 10)
        trames_traitees.append(np.fft.ifft(Env2 * (E/max(E)) * np.exp(1j * phase)).real)
    plt.plot(np.abs(Env), label='env_ce')
    plt.plot(np.abs(Env2), label='env2_ce')

    s = reconstruction_signal(trames_traitees, hop_len)

    #plt.figure()
    # plt.plot((s * raw.max()) + np.mean(raw))
    # plt.figure()
    # plt.plot(np.abs(T))
    # plt.plot(np.abs(Env))
    # plt.show()
    #x = s * raw.max() + np.mean(raw)
    # signal_rehausse = overlappedDFT[range(0, len(overlappedDFT), 2)]
    write_wav_file(s * 4, 'sound_files/fft.wav', fs)


    pass




def compressionSpectreCentree(signal, steps = 2):
    s = signal[range(0, len(signal), steps)]
    r = np.zeros(len(signal))
    half_s = int(len(s)/2)
    half_r = int(len(r)/2)
    start = half_r-half_s
    end = start + len(s)
    r[range(start, end)] = s
    return r

def compressionSpectre(signal, steps = 2):
    # tmp = X[range(0, len(X), 2)]
    # X2 = np.zeros(len(X))
    # X2[range(0, int(len(tmp) / 2))] = tmp[range(0, int(len(tmp) / 2))]
    # X2[range(len(X) - int(len(tmp) / 2), len(X))] = tmp[range(int(len(tmp) / 2) + 1, len(tmp))]
    s = signal[range(0, len(signal), steps)]
    r = np.zeros(len(signal))
    half_s = int(len(s)/2)
    i = len(s) - half_s - half_s
    r[range(0, half_s)] = s[range(0, half_s)]
    r[range(len(r)-half_s, len(r))] = s[range(half_s + i, len(s))]
    return r


def normalisation_signal(signal_in):
    return signal_in / max(signal_in)


def soustraire_moyenne(signal_in):
    return signal_in - np.mean(signal_in)


def extraction_du_signal(file_path: str):
    fs, signal = open_wav_file(file_path)

    plt.figure()
    plt.title("Signal original")
    plt.plot(signal)
    return fs, signal


def trame_and_windowing(signal, window, out, nb_window, window_len):
    for curr_window in range(0, nb_window):
        if window is None:
            window = np.hanning(window_len)
        s_windowed = []
        half_window = window_len // 2

        for i in range(0, window_len):
            s_windowed.append(window[i] * signal[curr_window * half_window + i])

        out.append(s_windowed)


def detrame_add_window(trame_table, signal_complet, window_len):
    for i in range(0, len(trame_table)-1):
        lab.add_overlapping_window(trame_table[i], trame_table[i+1], signal_complet, window_len)


def zero_padding_debut_fin(signal, length1, length2):
    return np.concatenate((np.zeros(length1), signal, np.zeros(length2)))


def calcule_longueur_trame(fs, trame_length_ms):
    return round(np.floor(fs*trame_length_ms/1000))


def calcule_nombre_trame(signal, longueur_trame):
    return math.floor((len(signal) - longueur_trame) / (longueur_trame // 2)) + 1


def enveloppeSpectrale(amplitude_spectre: np.array, k: int = 10):
    a = 20 * np.log10(amplitude_spectre)
    Y = np.fft.fft(a)
    phase = np.angle(Y)
    amp = np.abs(Y)
    indices = range(k, len(Y) - k + 2)
    amp[indices] = 0
    return 10 ** (np.real(np.fft.ifft(amp * np.exp(1j * phase)))/20)


def extractionFondamentales(amplitude_spectre: np.array, threshold: float = 0.5):
    fondamentales = amplitude_spectre > threshold
    return amplitude_spectre * fondamentales


def dontMessWithPhase(trame: np.array, amplitude_spectre: np.array):
    phase_spectre = np.angle(trame)
    return amplitude_spectre * np.exp(1j * phase_spectre)


def preparation_du_signal(fpath):
    fs, signal = extraction_du_signal(fpath)  # Extraire signal, normalisation
    signal = normalisation_signal(signal)
    signal = soustraire_moyenne(signal)

    longueur_trame = calcule_longueur_trame(fs, 20)

    nb_window = calcule_nombre_trame(signal, longueur_trame)
    hanning = np.hanning(longueur_trame)  # Create window

    signal = zero_padding_debut_fin(signal, longueur_trame // 2, longueur_trame // 2)

    windowed_signal = []
    trame_and_windowing(signal, hanning, windowed_signal, nb_window, longueur_trame)
    return fs, windowed_signal, longueur_trame


def rehaussementLPC():
    fs, raw = open_wav_file("sound_files/hel_fr2.wav")
    signal = soustraire_moyenne(normalisation_signal(raw))
    frame_len, hop_len = 882, 441
    windowed_frames = trammeur_fenetreur(signal, frame_len, hop_len)
    trames_traitees = []
    for i, trame in enumerate(windowed_frames):
        a = librosa.lpc(np.array(trame), 60)
        E_n = sc.lfilter(a, 1, trame)
        E_w = np.fft.fft(E_n)
        E_w_amp = np.abs(E_w)
        E_w_phase = np.angle(E_w)
        [_, enveloppe] = sc.freqz(1, a, frame_len//2)

        frequence_reponse_frequence = np.abs(np.fft.fft(enveloppe))
        frequence_reponse_frequence = normalisation_signal(frequence_reponse_frequence)
        coefficient_enveloppe = librosa.lpc(frequence_reponse_frequence, 25)
        [_, enveloppe_reponse_frequence] = sc.freqz(1, coefficient_enveloppe, hop_len)
        enveloppe_reponse_frequence = compressionSpectreDCT(enveloppe_reponse_frequence, 3)
        enveloppe_reponse_frequence = np.concatenate((enveloppe_reponse_frequence, np.flipud(enveloppe_reponse_frequence)))

        trames_traitees.append(np.fft.ifft(enveloppe_reponse_frequence * E_w_amp * np.exp(1j * np.angle(np.fft.fft(trame)))).real)

    s = reconstruction_signal(trames_traitees, hop_len)

    s = normalisation_signal(s)
    i = np.abs(s) > 0.01
    s = s * i

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.stem(np.linspace(0, np.pi * 2, len(signal)), normalisation_signal(np.abs(np.fft.fft(signal))), label='Signal original')
    ax2.stem(np.linspace(0, np.pi * 2, len(s)), normalisation_signal(np.abs(np.fft.fft(s))), label='Signal rehaussé')
    ax1.set_title('Réponse en fréquence du signal original')
    ax2.set_title('Réponse en fréquence du signal rehaussé')
    ax1.set_xlabel('Fréquence $\omega$ [rads]')
    ax1.set_ylabel('Amplitude')
    ax2.set_xlabel('Fréquence $\omega$ [rads]')
    ax2.set_ylabel('Amplitude')

    plt.figure()
    plt.plot(normalisation_signal(signal), label='Signal original')
    plt.plot(s, label='Signal rehaussé')
    plt.title('Comparaison du rehaussement')
    plt.xlabel('Échantillon')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.show()
    write_wav_file(s, 'sound_files/lpc_close_enough_v_eric.wav', fs)
    return "What is the airspeed velocity of an unladen swallow?"


def compression_sans_perte():  # QMF
    return 2


def rehaussement_du_signal(file_path: str):
    # Extraction du signal
    fs, signal_fenetre, longueur_trame = preparation_du_signal(file_path)
    # ---Acquisition des paramètres---

    # À faire pour chaque:
    # Analyse selon la technique
    # Extraction de paramètres (coefficient de transformée, coefficients de filtre prédicteurs, etc)
    # Modification des paramètres => Permet de retrouvée une ENVELOPPE SPECTRALE comprimée d'un facteur 2 à 3
    # Aucun changement sur la position des harmoniques du signal d'origine
    #

    # Par approche LPC : Modélisation de l'enveloppe à l'aide d'un filtre adaptatif à prédiction linéaire
    rehaussementLPC()
    # Par approche DFT/FFT : Décomposition fréquentielle
    # rehaussementDFT()      #fs, signal_fenetre, longueur_trame)
    # Par approche DCT : Décomposition fréquentielle
    # rehaussementDCT()
    # ---Acquisition des paramètres---

    return "app"


def culdecul():
    # Codage avec boucle de retroaction de bruit
    # Step by step
    input_signal = []
    output_signal = []
    quantization_error = []
    u_n = []
    uq_n = []
    window_size = 882
    order = 1
    alpha = 0.8
    # 1 - Frame by Frame (20 ms)
        # e[n] = s[n] - sortieLPC

    predicted_signal = librosa.lpc(input_signal, order)
    prediction_error = input_signal - predicted_signal

    # 2 - Echantillon par echantillon (loop alert)
        # Entree quantizer
            # u[n] = e[n] + f[n]  (ou f[n] est la sortie du noise feedback filter)
        # Entree noise feedback filter
            # q[n] = u[n] - uq[n]   (ou uq[n] est la sortie du quantizer]
    noise_feedback = librosa.lpc(quantization_error/alpha, order)
    u_n = prediction_error + noise_feedback
    quantization_error = u_n - uq_n

    # 3 - Frame by Frame (20 ms)
        # sq[n] = uq[n] + sortieLPC
    output_signal = uq_n + librosa.lpc(output_signal)
    return 2

def get_prediction_error(in_trame, ordre):
    """
    :param ordre: LPC filter order
    :param in_trame: 20 ms trame (ndarray)
    :return: 20 ms prediction error trame
    """
    out_trame = in_trame - librosa.lpc(in_trame, order=ordre)
    return out_trame

def get_output_signal(in_trame, ordre):
    """
    :param ordre: LPC filter order
    :param in_trame: 20 ms trame (ndarray) - Output of quantizer
    :return: 20 ms output signal trame
    """
    out_trame = in_trame + librosa.lpc(in_trame, order=ordre)
    return out_trame

def B_2():

    return 2

def quantizeLF():
    """
    Quantize low frequencies on X bits
    :return:
    """
    return 'notHighAF'

def quantizeHF():
    """
    Quantize high frequencies on X bits
    :return:
    """
    return 'highAF'
