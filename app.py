import math
import random
from typing import List

from numpy import ndarray
from scipy.io import wavfile
import scipy.signal as sc
import numpy as np
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


def rehaussementDFT(file_path: str):
    # Analyse selon la technique
    # Extraction de paramètres (coefficient de transformée, coefficients de filtre prédicteurs, etc)
    # Modification des paramètres => Permet de retrouvée une ENVELOPPE SPECTRALE comprimée d'un facteur 2 à 3
    # Aucun changement sur la position des harmoniques du signal d'origine
    fs, s = open_wav_file(file_path)
    s = s / max(s)
    n = range(0, len(s) - 1)
    n_plus = range(1, len(s))

    plt.plot(s[n], s[n_plus], '.')
    s2 = np.zeros((2, len(s) - 1))
    s2[0, :] = s[n]
    s2[1, :] = s[n_plus]

    T = np.dot((1 / np.sqrt(2)), np.array([[1, 1], [1, -1]]))
    X = np.dot(T, s2)
    plt.figure()
    plt.plot(X[0, :], X[1, :], '.')

    PSD = abs(X[0])  # POWAH SPECTRUM DENSITY
    spectre_de_lautisme = np.fft.fft(PSD)
    spectre2 = spectre_de_lautisme
    spectre2 = spectre_de_lautisme[np.linspace(0, 2 * np.pi, int(len(spectre_de_lautisme) / 2))]
    E = abs(np.fft.ifft(spectre2))
    plt.figure()
    plt.plot(E)

    # X2 = X * 2
    # plt.figure()
    # plt.plot(X2[0, :], X2[1, :], '.')
    #
    # X3 = X * 3
    # plt.figure()
    # plt.plot(X3[0, :], X3[1, :], '.')

    pass


def rehaussementDCT(file_path: str):
    pass

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
    Y = np.fft.fft(amplitude_spectre)
    indices = range(k, len(Y) - k + 2)
    Y[indices] = 0
    return np.real(np.fft.ifft(Y))


def extractionFondamentales(amplitude_spectre: np.array, threshold: float = 0.5):
    fondamentales = amplitude_spectre > threshold
    return amplitude_spectre * fondamentales


def dontFUCKwithPhase(full_spectre: np.array, amplitude_spectre: np.array):
    phase_spectre = np.angle(full_spectre)
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


def rehaussementLPC(fs, s_fenetre, longueur_trame):
    signal_fenetre = s_fenetre.copy()

    ordre_filtre_lpc = 20

    predicted_signal_enveloppe = []
    for i in range(0, len(signal_fenetre)):
        signal_freq = np.fft.fft(signal_fenetre[i])
        signal_freq = np.fft.fftshift(signal_freq)
        signal_abs = np.real(signal_freq)
        signal_abs = signal_abs / max(signal_abs)
        # posons notre filtre LPC valide et fonctionnel

        predicted_signal_enveloppe.append(lab.lpc_window(signal_abs, ordre_filtre_lpc, longueur_trame))

        #gossage sur les coefficients
        signal_comprimee = predicted_signal_enveloppe[i][range(0, len(predicted_signal_enveloppe[i]), 2)]
        signal_comprimee = zero_padding_debut_fin(signal_comprimee, len(signal_comprimee)//2, len(signal_comprimee)//2)
        predicted_signal_enveloppe[i] = signal_comprimee

        if i % 15 == 0:
            plt.figure()
            signal_freq = signal_freq / max(signal_freq)
            s = enveloppeSpectrale(np.abs(signal_freq), 100)
            plt.stem(np.abs(signal_freq))
            plt.plot(np.abs(s), 'b')
            plt.plot(predicted_signal_enveloppe[i], 'r')
            plt.legend(['Enveloppe spectrale du signal', 'Sortie LPC'])
        #     plt.plot(signal_abs)
        #     plt.plot(predicted_signal_enveloppe[i])
            # plt.plot(signal_freq)
            # s = enveloppeSpectrale(np.abs(signal_freq))
            # plt.plot(s)
        #     plt.figure()
        #     plt.title("i = " + str(i) + "m = " + str(ordre_filtre_lpc))
        #
        #     l1 = plt.plot(20*np.log10(signal_abs))
        #     l2 = plt.plot(20*np.log10(predicted_signal_enveloppe[i])+15, '-r')
        #     plt.legend([l1, l2], ['Signal', 'LPC'])

    corrected_predicted_signal_freq = []
    for curr_window in predicted_signal_enveloppe:
        corrected_predicted_signal_freq.append(curr_window[range(0, len(curr_window), 3)])

    conv_signal: list[ndarray] = []
    k = 0
    for w in signal_fenetre:
        ifft_filtre = np.fft.ifft(corrected_predicted_signal_freq[k])
        conv_signal.append(np.convolve(w, ifft_filtre))
        # if k < 10:
        #     plt.figure()
        #     plt.plot(conv_signal[k])
        k += 1

    overlapped = []
    detrame_add_window(conv_signal, overlapped, longueur_trame)

    overlapped = overlapped / max(overlapped)
    signal_rehausse = overlapped[range(0, len(overlapped), 2)]

    plt.figure()
    plt.title('Signal rehaussé')
    plt.plot(signal_rehausse)

    lab.write_wav_file(np.real(signal_rehausse), 'hel1_rehausse_3.wav', fs)

    # plt.plot(signal)

    # o_signal = np.fft.ifft(predicted_signal)
    # signal_added = []
    # for i in range(0, len(o_signal)-1):
    #     lab.add_overlapping_window(o_signal[i], o_signal[i+1], signal_added, window_len)
    #
    # signal_added = np.array(signal_added)
    # plt.figure()
    # plt.plot(signal_added)

    # Correction du facteur 2 - 3

    # ecriture fichier
    # lab.write_wav_file(np.abs(signal_added).astype(float), "sound_files\\o_signal.wav", fs)

    #
    # # nb_window = round(2 * len(signal)/window_len) - 1
    # nb_window = 3
    # windowed_signal = []
    # lab.overlap_filtering(signal, windowed_signal, nb_window, window_len, hanning)
    # for m in COEFFICIENTS:
    #     for s_fenetre in windowed_signal[:-1]:
    #         s_fenetre = np.abs(s_fenetre)
    #         a = librosa.lpc(np.array(s_fenetre), m)
    #         [w, H] = sc.freqz(1, a, half_window_len)
    #         Ha = np.abs(H)
    #
    #         fig, (ax1, ax2) = plt.subplots(1, 2)
    #         ax1.plot(w, np.abs(H))
    #         ax2.plot(w, s_fenetre[:half_window_len])

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
    rehaussementLPC(fs, signal_fenetre, longueur_trame)
    # Par approche DFT/FFT : Décomposition fréquentielle
    # rehaussementDFT(file_path)
    # Par approche DCT : Décomposition fréquentielle
    # rehaussementDCT(file_path)
    # ---Acquisition des paramètres---

    return "app"


def qmf_pb(file_path: str, upsampling: bool):
    """
    Low Pass an input signal and downscale or upsampling.
    :param file_path: Input signal
    :param upsampling: Wether or not the signal should be upsampled
    :return: Low passed and up/down sampled signal
    """
    cutoff_freq = 500
    coeff = [0.00938715, 0.06942827, -0.07065183, 0.48998080, 0.48998080, -0.07065183, 0.06942827, 0.00938715]
    h_pb = coeff
    h_ph = [((-1)**i)*coeff[i] for i in range(len(h_pb))]
    fs, input_signal = open_wav_file(file_path)
    #input_signal_freq = np.fft.fft(input_signal)
    signal_pb = sc.lfilter(h_pb, 1, input_signal)
    signal_pb_freq = np.fft.fft(signal_pb)
    signal_ph = sc.lfilter(h_ph, 1, input_signal)
    signal_ph_freq = np.fft.fft(signal_ph)
    #for n in range(len(input_signal)):
        #for j in range(len(coeff)):
            #if j < n:
                #signal_pb[n] = signal_pb[n] + (-1**n)*coeff[len(coeff)-1-j]*input_signal[n-j]
                #signal_ph[n] = signal_pb[n] + (-1 ** n) * coeff[len(coeff) - 1 - j] * input_signal[n - j]
    #signal_pb_freq = np.fft.fft(signal_pb)
    fig, (ax1) = plt.subplots(1, 1)
    ax1.plot(signal_ph_freq / max(signal_ph_freq), 'r')
    ax1.plot(signal_pb_freq/max(signal_pb_freq))

    ax1.set_title("Passe bas")
    #ax2.set_title("Passe haut")
    plt.show()
    #input_freq = np.fft.fft(input_signal)
    #Passe haut
    #Filtre passe-bas dans le livre de reference
    #filtered_signal = sc.lfilter(b, a, input_freq)
    #fig = plt.axes()
    #fig.stem(filtered_signal)
    return 2


def conventional_noise_feedback_coding(file_path: str):
    # Codage avec boucle de retroaction de bruit
    # Step by step
    fs, input_signal = open_wav_file(file_path)

    input_signal = input_signal/max(input_signal)
    input_signal = input_signal-np.mean(input_signal)
    output_signal = []
    quantization_error = []
    prediction_error = []
    window_size = 882
    order = 15
    alpha = 0.8
    u_n = 0     # Quantizer input (single sample)
    uq_n = 0    # Quantizer output (single sample)
    q_n = collections.deque(maxlen=order+1)    # Noise feedback Filter input (Sample buffer of 'order' length)
    f_n = [0]     # Noise feedback Filter output (single sample)
    # 1 - Frame by Frame (20 ms)
    for i in range(0, len(input_signal), window_size):
        trame = input_signal[i:i + window_size].astype('float')
        prediction_error, error_coefficients = get_prediction_error(trame, ordre=order)
        sq_n = []  # Output array
        #fig1 = plt.axes()
        #fig1.plot(trame)
        #fig1.plot(prediction_error, '--r', linewidth=1.0)
        #fig1.set_title('Erreur de prediction')
        #coefficients = librosa.lpc(prediction_error, order=order) / alpha
        quantize_range = get_quantize_range(prediction_error, 8)
        noise_feedback_coefficients = librosa.lpc(np.array(trame).astype('float'), order=order) #* alpha #Coefficients remain the same for a whole trame
        for k in range(len(noise_feedback_coefficients)):
            noise_feedback_coefficients[k] = noise_feedback_coefficients[k]*(alpha**k)
        # Avoir 'ordre' echantillons avant de calculer les coefficients du filtre de feedback.
        for j in range(len(prediction_error)):
            u_n = prediction_error[j] + f_n[0]
            uq_n = quantize_sample(u_n, quantize_range)
            #q_n = [u_n - uq_n]
            q_n.append(u_n - uq_n)
            if len(q_n) == q_n.maxlen:
                #Noise Feedback Filter
                f_n = sc.lfilter(noise_feedback_coefficients, 1, q_n)
            sq_n.append(uq_n)
        output_trame = get_output_trame(sq_n, coefficients=error_coefficients)
        fig = plt.axes()
        fig.plot(trame)
        fig.plot(output_trame, '--r', linewidth=1.0)
        fig.set_title('Trame de sortie')

    return 2


def get_noise_feedback(buffer, coefficients, ordre, constante):
    """
    Gets the noise feedback for a 'ordre' order filter
    :param coefficients: Filter coefficients
    :param buffer: Quantizer error buffer of length 'ordre' -> Quantizer_input - Quantizer_output
    :param ordre: Filter order
    :param constante: Value given in Annex C "Noise Feedback Coding"
    :return:
    """
    #error_coefficients = librosa.lpc(np.array(window).astype('float'), order=ordre) / constante
    #coefficients = -error_coefficients[1:]
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
    #Is the error simply the signal being predicted, or is it the difference -> in - predicted
    prediction_error_coefficients = librosa.lpc(in_trame, order=ordre)
    prediction_error = sc.lfilter(prediction_error_coefficients, 1, in_trame)
    #_, i_inverse_enveloppe = sc.freqz(1, prediction_error_coefficients, len(in_trame)//2)
    #i_inverse_enveloppe = i_inverse_enveloppe/max(i_inverse_enveloppe)
    #fft_trame = np.fft.fft(in_trame)
    #fft_trame = fft_trame/max(fft_trame)
    #plt.figure()
    #plt.plot(np.abs(fft_trame))
    #plt.plot(i_inverse_enveloppe, '--r', linewidth=1.0)
    #prediction_error = in_trame - predicted_signal
    return prediction_error, prediction_error_coefficients


def get_output_trame(in_trame, coefficients):
    """
    :param coeffcients: First LPC Filter coefficients (prediction error coefficients)
-    :param in_trame: 20 ms trame (ndarray) - Output of quantizer
    :return: 20 ms output signal trame
    """
    #prediction_error_coefficients = librosa.lpc(np.array(in_trame), order=ordre)
    #prediction_error_coefficients = -prediction_error_coef+ficients[1:]
    #out_trame = sc.lfilter(1, coefficients, in_trame) #TODO: FIX THIS
    out_trame = sc.lfilter(coefficients, 1, in_trame)  # TODO: FIX THIS
    #FFT SIGNAL -> DOMAIN FREQUENTIEL
    #FREQZ avec Coefficients erreur
    #Multiplication entre signal en frequence et reponse en frequences du filtre.
    #in_trame_freq = np.fft.fft(in_trame)
    #_, half_inverse_filter_freq_response = sc.freqz(b=1, a=prediction_error_coefficients, worN=len(in_trame_freq)//2)
    #inverse_filter_freq_response = np.concatenate((half_inverse_filter_freq_response, np.flipud(half_inverse_filter_freq_response)))
    #fig = plt.figure()
    #fig.stem(in_trame_freq)

    #fig, (ax1, ax2) = plt.subplots(1, 2)
    #ax1.stem(in_trame_freq)
    #ax2.stem(inverse_filter_freq_response)
    #ax1.legend(['Input'])
    #ax2.legend(['Fitler reponse'])
    #plt.show()

    #out_trame_freq = in_trame_freq * inverse_filter_freq_response
    #out_trame = np.real(np.fft.ifft(out_trame_freq))
    #out_trame = in_trame + predicted_signal
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
