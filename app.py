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
import collections
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

def test_sinus(f):
    """
    tests the first station algo
    :param f: the desired frequency
    """
    fs = 44100
    sample = 44100
    x = np.arange(sample)
    signal = 2 * np.sin(2 * np.pi * f * x / fs)
    signal2 = 2 * np.sin(2 * np.pi * f * 2 * x / fs)

    # noise = 0.5 * np.sin(2 * np.pi * f // 2 * x / fs)
    # noise += 0.25 * np.sin(2 * np.pi * f ** 2 * x / fs)
    # noise = np.random.normal(0, 1, sample)
    # signal = signal + noise
    signal = signal + signal2
    return signal

def rehaussementLPC():
    fs, raw = open_wav_file("sound_files/hel_fr2.wav")
    signal = soustraire_moyenne(normalisation_signal(raw))
    frame_len, hop_len = 882, 441
    windowed_frames = trammeur_fenetreur(signal, frame_len, hop_len)
    # s = reconstruction_signal(windowed_frames, hop_len)
    trames_traitees = []
    for i, trame in enumerate(windowed_frames):
        a = librosa.lpc(np.array(trame), 350)
        E_n = sc.lfilter(a, 1, trame)
        E_w = np.fft.fft(E_n)
        E_w_amp = np.abs(E_w)
        E_w_phase = np.angle(E_w)

        [fr, enveloppe] = sc.freqz(1, a, frame_len//2)
        env_amp = np.abs(enveloppe)
        env_phase = np.angle(enveloppe)

        enveloppe_flip = np.concatenate((enveloppe, np.flipud(enveloppe)))

        # enveloppe = enveloppeSpectrale(np.abs(E_w), 50)
        # E = np.abs(E_w) / enveloppe
        enveloppe_downsampling = compressionSpectre(enveloppe_flip, 3)
        # plt.figure()
        # plt.stem(enveloppe_downsampling)
        # plt.plot(E_w)
        trames_traitees.append(np.fft.ifft(enveloppe_downsampling * E_w_amp * np.exp(1j * E_w_phase)).real)

    # plt.figure()
    # plt.title('down')
    # plt.plot(E_n, )
    # plt.plot(E_w)
    # plt.plot(enveloppe_flip)

    # plt.plot(enveloppe, label='env_gath')
    # plt.plot(enveloppe_downsampling, )
    # rehaussementDFT()

    plt.legend()
    plt.show()
    s = reconstruction_signal(trames_traitees, hop_len)

    #debruitage
    s = normalisation_signal(s)
    i = np.abs(s) > 0.005
    s = s * i

    # plt.figure()
    # plt.plot((s * raw.max()) + np.mean(raw))
    # plt.figure()
    # plt.plot(np.abs(T))
    # plt.plot(np.abs(Env))
    # x = s * raw.max() + np.mean(raw)
    # signal_rehausse = overlappedDFT[range(0, len(overlappedDFT), 2)]
    write_wav_file(s, 'sound_files/lpc.wav', fs)
    plt.show()
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


def qmf(file_path: str):
    """
    Splits signal in two equal bands before coding it
    :param file_path: Input signal
    :return: Synthesized signal
    """
    #Coefficients du livre de reference 16-TAP
    coeff = [0.002898163, -0.009972252, -0.001920936, 0.03596853, -0.01611869, -0.09530234, 0.1067987, 0.4773469,
             0.4773469, 0.1067987, -0.09530234, -0.01611869, 0.03596853, -0.001920936, -0.009972252, 0.002898163]
    # Low-Pass filter
    h_pb = coeff
    # High-pass filter coefficients
    h_ph = [((-1)**i)*coeff[i] for i in range(len(h_pb))]
    fs, input_signal = open_wav_file(file_path)
    # Low pass signal
    signal_pb = sc.lfilter(h_pb, 1, input_signal)
    signal_pb_freq = np.fft.fft(signal_pb)
    # High pass signal
    signal_ph = sc.lfilter(h_ph, 1, input_signal)
    signal_ph_freq = np.fft.fft(signal_ph)
    # Down sample each band -> Keep every 1/2 sample
    signal_pb_downsampled = signal_pb[range(0, len(signal_pb), 2)]
    signal_ph_downsampled = signal_ph[range(0, len(signal_ph), 2)]
    # Synthetize signal
    signal_pb_synth = conventional_noise_feedback_coding(signal_pb_downsampled, 3)
    signal_ph_synth = conventional_noise_feedback_coding(signal_ph_downsampled, 3)
    #TODO:Upsampling
    signal_pb_synth = upsample(signal_pb_synth)
    signal_ph_synth = upsample(signal_ph_synth)
    # 2nd and final filtering
    signal_pb_pb_synth = sc.lfilter(h_pb, 1, signal_pb_synth)
    signal_ph_ph_synth = sc.lfilter(h_ph, 1, signal_ph_synth)
    # Output signal
    output_signal = signal_pb_pb_synth + signal_ph_ph_synth
    write_wav_file(output_signal, 'sound_files/qmf.wav', fs)
    return 2


def upsample(buffer):
    """
    :param buffer: Buffer to upsamle
    :return: upsampled buffer
    """
    result = [0] * (len(buffer) * 2 - 1)
    result[0::2] = buffer
    return result


def conventional_noise_feedback_coding(in_signal, n_bits):
    """
    Code signal using convetional noise feedback
    :param in_signal: input signal to code
    :param n_bits: number of bits to quantize signal
    :return: Output signal
    """
    input_signal = in_signal
    input_signal = input_signal/max(input_signal)
    input_signal = input_signal-np.mean(input_signal)
    output_signal = []
    prediction_error = []
    window_size = 882
    order = 15
    alpha = 0.8
    u_n = 0     # Quantizer input (single sample)
    uq_n = 0    # Quantizer output (single sample)
    q_n = collections.deque(maxlen=order+1)    # Quantization error -> Noise feedback Filter input
    f_n = [0]     # Noise feedback Filter output (single sample)
    # 1 - Frame by Frame (20 ms)
    for i in range(0, len(input_signal), window_size):
        trame = input_signal[i:i + window_size].astype('float')
        prediction_error, error_coefficients = get_prediction_error(trame, ordre=order)
        sq_n = []  # Output array
        quantize_range = get_quantize_range(prediction_error, n_bits)
        noise_feedback_coefficients = librosa.lpc(np.array(trame).astype('float'), order=order) #Coefficients remain the same for a whole trame
        for k in range(len(noise_feedback_coefficients)):
            noise_feedback_coefficients[k] = noise_feedback_coefficients[k]*(alpha**k)
        # Avoir 'ordre' echantillons avant de filtrer l'erreur de quantification
        # 2 - Sample by Sample
        for j in range(len(prediction_error)):
            u_n = prediction_error[j] + f_n[0]
            uq_n = quantize_sample(u_n, quantize_range)
            q_n.append(u_n - uq_n)
            if len(q_n) == q_n.maxlen:
                #Noise Feedback Filter
                f_n = sc.lfilter(noise_feedback_coefficients, 1, q_n)
            sq_n.append(uq_n)
        output_trame = get_output_trame(sq_n, coefficients=error_coefficients)
        output_signal.extend(output_trame)
        #fig = plt.axes()
        #fig.plot(prediction_error, label='Erreur de prédiction')
        #fig.plot(sq_n, '--r', linewidth=1.0, label='Erreur de prédiction quantifiée')
        #fig.set_title('Quantificateur')
        #fig.set_xlabel('n')
        #fig.set_ylabel('Amplitude')
        #fig.legend()
    return output_signal


def get_noise_feedback(buffer, coefficients, ordre, constante):
    """
    Gets the noise feedback for a 'ordre' order filter
    :param coefficients: Filter coefficients
    :param buffer: Quantizer error buffer of length 'ordre' -> Quantizer_input - Quantizer_output
    :param ordre: Filter order
    :param constante: Value given in Annex C "Noise Feedback Coding"
    :return:
    """
    noise_feedback = sc.lfilter(coefficients, 1, buffer)
    return noise_feedback


def get_quantize_range(in_trame, n_bits):
    """
    Returns the possible values of quantization with n_bits
    :param in_trame: Trame to use
    :param n_bits: Number of bits
    :return: array of possible values when quantizing bits
    """
    in_min = np.min(in_trame)
    in_max = np.max(in_trame)
    out_possible_values = np.linspace(start=in_min, stop=in_max, num=((2**n_bits) - 1))
    return out_possible_values


def get_prediction_error(in_trame, ordre):
    """
    :param ordre: LPC filter order
    :param in_trame: 20 ms trame (ndarray)
    :return: 20 ms prediction error trame and prediction error coefficients.
    """
    prediction_error_coefficients = librosa.lpc(in_trame, order=ordre)
    prediction_error = sc.lfilter(prediction_error_coefficients, 1, in_trame)
    return prediction_error, prediction_error_coefficients


def get_output_trame(in_trame, coefficients):
    """
    :param coeffcients: First LPC Filter coefficients (prediction error coefficients)
-    :param in_trame: 20 ms trame (ndarray) - Output of quantizer
    :return: 20 ms output signal trame
    """
    #out_trame = sc.lfilter(1, coefficients, in_trame) #TODO: FIX THIS. THIS SHOULD WORK, BUT IT DOESN'T. MUCH HAPPINESS
    out_trame = sc.lfilter(coefficients, 1, in_trame)  # Th
    return out_trame


def quantize_sample(in_sample, quantize_range):
    """
    Quantize low frequencies on according to a list of possible values
    :param in_sample: input sample
    :param quantize_range: possible output values
    :return: the quantized sample
    """
    idx = (np.abs(quantize_range - in_sample)).argmin()
    quantized_sample = quantize_range[idx]
    return quantized_sample
