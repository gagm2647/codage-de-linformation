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
    # s = reconstruction_signal(windowed_frames, hop_len)
    trames_traitees = []
    for i, trame in enumerate(windowed_frames):
        T = scfft.dct(trame)
        neg = T < 0
        Env = enveloppeSpectreDCT(abs(T), 45)
        E = T / Env
        Env2 = compressionSpectreDCT(Env, 3)

        trames_traitees.append(scfft.idct(Env2 * E))

    s = reconstruction_signal(trames_traitees, hop_len)

    write_wav_file(s, 'sound_files/dct.wav', fs)

    return "What is your name? Arthur Kings of the Britons. What is your quest? To seek the grail! What is the air " \
           "speed velocity of an unladen swallow? What do you mean? african or european? Huh, I don't know that?! "


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
    # s = reconstruction_signal(windowed_frames, hop_len)
    trames_traitees = []
    for i, trame in enumerate(windowed_frames):
        # b, a = sc.butter(3, 10000/fs, btype='low', output='ba')
        # trame = sc.filtfilt(b, a, trame)  # trame[range(hop_len, frame_len)] = 0
        # print(trame)
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
        # E2 = enveloppeSpectrale(np.abs(T2), 10)
        trames_traitees.append(np.fft.ifft(Env2 * (E / max(E)) * np.exp(1j * phase)).real)
    plt.plot(np.abs(Env), label='env_ce')
    plt.plot(np.abs(Env2), label='env2_ce')

    s = reconstruction_signal(trames_traitees, hop_len)

    # plt.figure()
    # plt.plot((s * raw.max()) + np.mean(raw))
    # plt.figure()
    # plt.plot(np.abs(T))
    # plt.plot(np.abs(Env))
    # plt.show()
    # x = s * raw.max() + np.mean(raw)
    # signal_rehausse = overlappedDFT[range(0, len(overlappedDFT), 2)]
    write_wav_file(s * 4, 'sound_files/fft.wav', fs)

    pass


def compressionSpectreCentree(signal, steps=2):
    s = signal[range(0, len(signal), steps)]
    r = np.zeros(len(signal))
    half_s = int(len(s) / 2)
    half_r = int(len(r) / 2)
    start = half_r - half_s
    end = start + len(s)
    r[range(start, end)] = s
    return r


def compressionSpectre(signal, steps=2):
    # tmp = X[range(0, len(X), 2)]
    # X2 = np.zeros(len(X))
    # X2[range(0, int(len(tmp) / 2))] = tmp[range(0, int(len(tmp) / 2))]
    # X2[range(len(X) - int(len(tmp) / 2), len(X))] = tmp[range(int(len(tmp) / 2) + 1, len(tmp))]
    s = signal[range(0, len(signal), steps)]
    r = np.zeros(len(signal))
    half_s = int(len(s) / 2)
    i = len(s) - half_s - half_s
    r[range(0, half_s)] = s[range(0, half_s)]
    r[range(len(r) - half_s, len(r))] = s[range(half_s + i, len(s))]
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
    for i in range(0, len(trame_table) - 1):
        lab.add_overlapping_window(trame_table[i], trame_table[i + 1], signal_complet, window_len)


def zero_padding_debut_fin(signal, length1, length2):
    return np.concatenate((np.zeros(length1), signal, np.zeros(length2)))


def calcule_longueur_trame(fs, trame_length_ms):
    return round(np.floor(fs * trame_length_ms / 1000))


def calcule_nombre_trame(signal, longueur_trame):
    return math.floor((len(signal) - longueur_trame) / (longueur_trame // 2)) + 1


def enveloppeSpectrale(amplitude_spectre: np.array, k: int = 10):
    a = 20 * np.log10(amplitude_spectre)
    Y = np.fft.fft(a)
    phase = np.angle(Y)
    amp = np.abs(Y)
    indices = range(k, len(Y) - k + 2)
    amp[indices] = 0
    return 10 ** (np.real(np.fft.ifft(amp * np.exp(1j * phase))) / 20)


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
    signal = soustraire_moyenne(normalisation_signal(raw))  # pretraitement du signal (normalisation retrait de la dc)
    frame_len, hop_len = 882, 441
    windowed_frames = trammeur_fenetreur(signal, frame_len, hop_len)  # Fenetrage par methode Overlap-Add / Hanning

    trames_traitees = []
    for i, trame in enumerate(windowed_frames):
        a = librosa.lpc(np.array(trame), 60)  # Extraction des coefficients de l'erreur
        E_n = sc.lfilter(a, 1, trame)  # Filtrage du signal pour obtenir l'excitation
        E_w = np.fft.fft(E_n)  # composante frequentielle de l'excitation
        E_w_amp = np.abs(E_w)
        E_w_phase = np.angle(E_w)
        [_, enveloppe] = sc.freqz(1, a, frame_len // 2)  # Extraction de l'enveloppe

        frequence_reponse_frequence = np.abs(np.fft.fft(enveloppe))  # FFT de l'enveloppe
        frequence_reponse_frequence = normalisation_signal(frequence_reponse_frequence)  # Normalisation
        coefficient_enveloppe = librosa.lpc(frequence_reponse_frequence, 25)  # Coefficient de l'erreur de l'enveloppe
        [_, enveloppe_reponse_frequence] = sc.freqz(1, coefficient_enveloppe, hop_len)  # Enveloppe de la reponse en frequence
        enveloppe_reponse_frequence = compressionSpectreDCT(enveloppe_reponse_frequence, 3)  # Compression du spectre
        enveloppe_reponse_frequence = np.concatenate((enveloppe_reponse_frequence, np.flipud(enveloppe_reponse_frequence)))  # Sur 0 2pi complet

        trames_traitees.append(np.fft.ifft(enveloppe_reponse_frequence * E_w_amp * np.exp(1j * np.angle(np.fft.fft(trame)))).real)  # Application de l'enveloppe sur l'excitation

    s = reconstruction_signal(trames_traitees, hop_len)  # Defenetrage

    s = normalisation_signal(s)
    # Debruitage
    i = np.abs(s) > 0.01
    s = s * i

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.stem(np.linspace(0, np.pi * 2, len(signal)), normalisation_signal(np.abs(np.fft.fft(signal))),
             label='Signal original')
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

def rehaussement_du_signal(file_path: str):
    rehaussementLPC()
    rehaussementDFT()
    rehaussementDCT()

    return "First shalt thou take out the Holy Pin. Then shalt thou count to three, no more, no less. Three shall be " \
           "the number thou shalt count, and the number of the counting shall be three. Four shalt thou not count, " \
           "neither count thou two, excepting that thou then proceed to three. "


def qmf(file_path: str):
    """
    Splits signal in two equal
    :param file_path: Input signal
    :return: Synthesized signal
    """
    # coeff = [0.00938715, 0.06942827, -0.07065183, 0.48998080, 0.48998080, -0.07065183, 0.06942827, 0.00938715]
    coeff = [0.002898163, -0.009972252, -0.001920936, 0.03596853, -0.01611869, -0.09530234, 0.1067987, 0.4773469,
             0.4773469, 0.1067987, -0.09530234, -0.01611869, 0.03596853, -0.001920936, -0.009972252, 0.002898163]
    h_pb = coeff
    h_ph = [((-1) ** i) * coeff[i] for i in range(len(h_pb))]
    fs, input_signal = open_wav_file(file_path)
    signal_pb = sc.lfilter(h_pb, 1, input_signal)
    signal_pb_freq = np.fft.fft(signal_pb)
    signal_ph = sc.lfilter(h_ph, 1, input_signal)
    signal_ph_freq = np.fft.fft(signal_ph)
    fig, (ax1) = plt.subplots(1, 1)
    ax1.plot(np.linspace(0, np.pi * 2, len(signal_ph_freq)), signal_ph_freq / max(signal_ph_freq), 'r',
             label='Passe-haut')
    ax1.plot(np.linspace(0, np.pi * 2, len(signal_pb_freq)), signal_pb_freq / max(signal_pb_freq), label='Passe-bas')
    ax1.set_ylabel('Amplitude')
    ax1.set_xlabel('Fréquence $\omega$ [rads]')
    ax1.set_title("Signal séparé en sous-bandes")
    # ax2.set_title("Passe haut")
    plt.legend()
    plt.show()
    # Down sampling
    signal_pb_downsampled = signal_pb[range(0, len(signal_pb), 2)]
    signal_ph_downsampled = signal_pb[range(0, len(signal_ph), 2)]
    # Synthetize signal
    signal_pb_synth = conventional_noise_feedback_coding(signal_pb_downsampled, 3)
    signal_ph_synth = conventional_noise_feedback_coding(signal_ph_downsampled, 3)
    return "If she weighs the same as a duck... she's made of wood!"


def conventional_noise_feedback_coding(in_signal, n_bits):
    # Codage avec boucle de retroaction de bruit
    # Step by step
    # fs, input_signal = open_wav_file(file_path)
    input_signal = in_signal
    input_signal = input_signal / max(input_signal)
    input_signal = input_signal - np.mean(input_signal)
    output_signal = []
    quantization_error = []
    prediction_error = []
    window_size = 882
    order = 15
    alpha = 0.8
    u_n = 0  # Quantizer input (single sample)
    uq_n = 0  # Quantizer output (single sample)
    q_n = collections.deque(maxlen=order + 1)  # Noise feedback Filter input (Sample buffer of 'order' length)
    f_n = [0]  # Noise feedback Filter output (single sample)
    # 1 - Frame by Frame (20 ms)
    for i in range(0, len(input_signal), window_size):
        trame = input_signal[i:i + window_size].astype('float')
        prediction_error, error_coefficients = get_prediction_error(trame, ordre=order)
        sq_n = []  # Output array
        # fig1 = plt.axes()
        # fig1.plot(trame)
        # fig1.plot(prediction_error, '--r', linewidth=1.0)
        # fig1.set_title('Erreur de prediction')
        # coefficients = librosa.lpc(prediction_error, order=order) / alpha
        quantize_range = get_quantize_range(prediction_error, n_bits)
        noise_feedback_coefficients = librosa.lpc(np.array(trame).astype('float'),
                                                  order=order)  # * alpha #Coefficients remain the same for a whole trame
        for k in range(len(noise_feedback_coefficients)):
            noise_feedback_coefficients[k] = noise_feedback_coefficients[k] * (alpha ** k)
        # Avoir 'ordre' echantillons avant de calculer les coefficients du filtre de feedback.
        for j in range(len(prediction_error)):
            u_n = prediction_error[j] + f_n[0]
            uq_n = quantize_sample(u_n, quantize_range)
            # q_n = [u_n - uq_n]
            q_n.append(u_n - uq_n)
            if len(q_n) == q_n.maxlen:
                # Noise Feedback Filter
                f_n = sc.lfilter(noise_feedback_coefficients, 1, q_n)
            sq_n.append(uq_n)
        output_trame = get_output_trame(sq_n, coefficients=error_coefficients)
        fig = plt.axes()
        fig.plot(prediction_error, label='Erreur de prédiction')
        fig.plot(sq_n, '--r', linewidth=1.0, label='Erreur de prédiction quantifiée')
        fig.set_title('Quantificateur')
        fig.set_xlabel('n')
        fig.set_ylabel('Amplitude')
        fig.legend()

    return "We are the Knights who say.....NI!"


def get_noise_feedback(buffer, coefficients, ordre, constante):
    """
    Gets the noise feedback for a 'ordre' order filter
    :param coefficients: Filter coefficients
    :param buffer: Quantizer error buffer of length 'ordre' -> Quantizer_input - Quantizer_output
    :param ordre: Filter order
    :param constante: Value given in Annex C "Noise Feedback Coding"
    :return:
    """
    # error_coefficients = librosa.lpc(np.array(window).astype('float'), order=ordre) / constante
    # coefficients = -error_coefficients[1:]
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
    out_possible_values = np.linspace(start=in_min, stop=in_max, num=((2 ** n_bits) - 1))
    return out_possible_values


def get_prediction_error(in_trame, ordre):
    """
    :param ordre: LPC filter order
    :param in_trame: 20 ms trame (ndarray)
    :return: 20 ms prediction error trame and prediction error coefficients.
    """
    # Is the error simply the signal being predicted, or is it the difference -> in - predicted
    prediction_error_coefficients = librosa.lpc(in_trame, order=ordre)
    prediction_error = sc.lfilter(prediction_error_coefficients, 1, in_trame)
    # _, i_inverse_enveloppe = sc.freqz(1, prediction_error_coefficients, len(in_trame)//2)
    # i_inverse_enveloppe = i_inverse_enveloppe/max(i_inverse_enveloppe)
    # fft_trame = np.fft.fft(in_trame)
    # fft_trame = fft_trame/max(fft_trame)
    # plt.figure()
    # plt.plot(np.abs(fft_trame))
    # plt.plot(i_inverse_enveloppe, '--r', linewidth=1.0)
    # prediction_error = in_trame - predicted_signal
    return prediction_error, prediction_error_coefficients


def get_output_trame(in_trame, coefficients):
    """
    :param coeffcients: First LPC Filter coefficients (prediction error coefficients)
-    :param in_trame: 20 ms trame (ndarray) - Output of quantizer
    :return: 20 ms output signal trame
    """
    # prediction_error_coefficients = librosa.lpc(np.array(in_trame), order=ordre)
    # prediction_error_coefficients = -prediction_error_coef+ficients[1:]
    # out_trame = sc.lfilter(1, coefficients, in_trame) #TODO: FIX THIS
    out_trame = sc.lfilter(coefficients, 1, in_trame)  # TODO: FIX THIS
    # FFT SIGNAL -> DOMAIN FREQUENTIEL
    # FREQZ avec Coefficients erreur
    # Multiplication entre signal en frequence et reponse en frequences du filtre.
    # in_trame_freq = np.fft.fft(in_trame)
    # _, half_inverse_filter_freq_response = sc.freqz(b=1, a=prediction_error_coefficients, worN=len(in_trame_freq)//2)
    # inverse_filter_freq_response = np.concatenate((half_inverse_filter_freq_response, np.flipud(half_inverse_filter_freq_response)))
    # fig = plt.figure()
    # fig.stem(in_trame_freq)

    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.stem(in_trame_freq)
    # ax2.stem(inverse_filter_freq_response)
    # ax1.legend(['Input'])
    # ax2.legend(['Fitler reponse'])
    # plt.show()

    # out_trame_freq = in_trame_freq * inverse_filter_freq_response
    # out_trame = np.real(np.fft.ifft(out_trame_freq))
    # out_trame = in_trame + predicted_signal
    return out_trame


def B_2():
    return 2


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
