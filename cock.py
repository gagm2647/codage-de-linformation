import math
import random
from scipy.io import wavfile
import scipy.signal as sc
import numpy as np
import matplotlib.pyplot as plt
import librosa


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


def rehaussementDFT(file_path:str):
    # Analyse selon la technique
    # Extraction de paramètres (coefficient de transformée, coefficients de filtre prédicteurs, etc)
    # Modification des paramètres => Permet de retrouvée une ENVELOPPE SPECTRALE comprimée d'un facteur 2 à 3
    # Aucun changement sur la position des harmoniques du signal d'origine
    fs, signal = open_wav_file(file_path)
    signal = signal / max(signal)  # Normalisation du signal
    n = len(signal)

    ret_val = plt.xcorr(signal.astype('float'), signal.astype('float'))
    _autocorrelation = ret_val[1]
    plt.figure()
    plt.plot(signal[:-1], signal[1:], 'ro')
    plt.figure()
    plt.plot(_autocorrelation[:-1], _autocorrelation[1:], 'ro')
    plt.show()
    COEFFICIENTS = [10, 20, n]
    # fenetres temporelles
    for c in COEFFICIENTS:
        #colors = np.linspace(start=100, stop=255, num=c)
        fft_signal = np.fft.fft(signal)
        plt.plot(abs(fft_signal))
        #for i in range(c):
         #

    pass


def rehaussementDCT(file_path:str):
    pass


def compression_sans_perte():  # QMF
    return 2


def rehaussement_du_signal(file_path: str):
    # Extraction du signal

    # ---Acquisition des paramètres---

    # À faire pour chaque:
    # Analyse selon la technique
    # Extraction de paramètres (coefficient de transformée, coefficients de filtre prédicteurs, etc)
    # Modification des paramètres => Permet de retrouvée une ENVELOPPE SPECTRALE comprimée d'un facteur 2 à 3
    # Aucun changement sur la position des harmoniques du signal d'origine
    #

    # Par approche LPC : Modélisation de l'enveloppe à l'aide d'un filtre adaptatif à prédiction linéaire

    # Par approche DFT/FFT : Décomposition fréquentielle
    rehaussementDFT(file_path)
    # Par approche DCT : Décomposition fréquentielle
    rehaussementDCT(file_path)
    # ---Acquisition des paramètres---

    return "app"

