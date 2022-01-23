from math import floor
from pretty_midi import PrettyMIDI
from numpy import ndarray as nparray, float32 as float32, power as pow, floor
from ComplexExponential import *
from pathlib import Path
import scipy.io.wavfile as wavfile
from playsound import playsound as play


def cosine(fs, T, f):
    """Generates the discrete cosine for the given input
    
    Parameters:
    :param fs: Sampling frequency
    T: time duration
    f: The requested frequency
    
    Returns:
    :return (cpx_cos, N)
    cpx_cos: Normalized complex exponential values
    N: Number of samples
    """
    # Not done in 1 line for readability only
    N: int = floor(T * fs)
    k = N * f / fs
    cpx_exp = ComplexExponential(k, N, Normalized=True)
    # IDK why we need to return N since it's in the object
    return (cpx_exp.e_kN.real, N)


def write_and_play(file: Path, f_samp: int, samps: nparray):
    """Generates a .WAV file at a specific path and plays it

    :param file: Path to where the soundfile should go
    :param f_samp sampling frequency
    :param samps array of samples of the audio"""
    wavfile.write(file.absolute(), f_samp, samps.astype(float32))
    play(str(file.absolute()))


def midi_number_to_piano_key(n):
    """Converts the number of a midi note to the piano key number"""
    # Data checks, but who needs those?? assert (n >= 21).all() and (n <= 108).all() and (floor(n) == n).all()
    return n - 20


def piano_key_to_frequency(n):
    # Data checks, but who needs those? assert (floor(n) == n).all() and (1 <= n).all() and (88 >= n).all()
    return pow(2, (n - 49) / 12) * 440

def read_midi_to_keys(file : Path):
    midi = PrettyMIDI(str(file))
    Ts = list()
    keys = list()
    delays = list()
    last_end = 0
    for note in midi.instruments[0].notes:
        Ts.append(note.duration)
        delays.append(max(0, note.start - last_end))
        last_end = note.end
        keys.append(midi_number_to_piano_key(note.pitch))
    return (keys, Ts, delays)