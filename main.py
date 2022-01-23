#necessary import statements for the code
import sys

import mido
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from music import *
from ComplexExponential import *
from pathlib import Path
from util import print_matrix

outdir = Path('output/')
if __name__ == "__main__":
    try:
        outdir.mkdir(parents=True, exist_ok=False)
    except:
        import os,shutil
        for files in os.listdir(outdir):
            path = os.path.join(outdir, files)
            try:
                shutil.rmtree(path)
            except OSError:
                os.remove(path)

def make_qdir(qN):
    qdir = Path(outdir, qN)
    qdir.mkdir(parents=True, exist_ok=True)
    return qdir

def save_figs(qN, figs):
    qdir = make_qdir(qN)
    for fig in figs:
        fig[0].savefig(Path(qdir, f'{fig[1]}.png'))

def plot_figure(ax : plt.Axes, k, N):
    # Creates complex exponential object with frequency k and duration N
    exp_k = ComplexExponential(k, N)
    # Real and imaginary parts
    cpx_cos = exp_k.e_kN.real
    cpx_sin = exp_k.e_kN.imag
    # Plots real and imaginary parts

    # plt.plot(exp_k.ns, cpx_cos, c='b', marker='o')
    # plt.plot(exp_k.ns, cpx_sin, c='g', marker='o')

    ax.stem(exp_k.ns, cpx_cos, 'tab:blue', markerfmt='bo', label='Real')
    ax.stem(exp_k.ns, cpx_sin, 'tab:green', markerfmt='go', label='Imaginary')

    # Labels, title
    ax.set_xlabel("n")
    ax.set_ylabel("x(n)")
    ax.set_title(f"Complex Exponential: k={k} N={N}")
    ax.figure.legend()
    ax.figure.show()

def plot_figures(N, ks):
    figures = list()
    for km in ks:
        # Create figure and axes
        cpx_fig = plt.figure()
        cpx_ax = cpx_fig.add_subplot()
        # Do the plotting
        plot_figure(cpx_ax, km, N)
        # Add to list so we can save later
        figures.append( (cpx_fig, f'k{km}n{N}') )
    return figures

#%%
#****** STILL NOT SURE WHY ITS NAMED Q 11 *********
# takes in a list of frequencies as and iterates through the frequencies in the list
def q_11(N, k_list):
    return plot_figures(N, k_list)

if __name__ == '__main__':
    list_of_ks = [0, 2, 9, 16]
    duration_of_signal = 32
    save_figs('q_11', q_11(duration_of_signal, list_of_ks))


def q_12(N, k):
    return plot_figures(N, [k - N, k, k + N])


if __name__ == '__main__':
    k = 3
    duration_of_signal = 32
    save_figs('q_12', q_12(duration_of_signal, k))


# %%
def q_13(N, k):
    return plot_figures(N, [k, -k])


if __name__ == '__main__':
    k = 3
    duration_of_signal = 32
    save_figs('q_13', q_13(duration_of_signal, k))


# %%
def q_14(N, k):
    return plot_figures(N, [k, N - k])


if __name__ == '__main__':
    k = 3
    duration_of_signal = 32
    save_figs('q_14', q_14(duration_of_signal, k))


# %%
def q_15(N):
    ks = np.arange(N)
    cpx_exps = np.zeros((N, N), dtype=complex)
    for k in ks:
        cpx_exp = ComplexExponential(k, N)
        cpx_exps[:, k] = cpx_exp.e_kN

    cpx_exps_conj = np.conjugate(cpx_exps)

    # Option 1: computing inner products simultaneously
    # TODO: norm, real, abs, or none??
    res = np.matmul(cpx_exps_conj, cpx_exps)
    print("\n Matrix of inner products: Mp =")
    print_matrix(res)
    fig = plt.matshow(np.abs(res)).figure

    return (fig, res)

if __name__ == '__main__':
    duration_of_signal = 16
    (fig, mp) = q_15(duration_of_signal)
    qfile_base = str(Path(make_qdir('q_15'), f'N{duration_of_signal}'))
    scipy.io.savemat(f'{qfile_base}.mat', {"N": duration_of_signal, "M_p": mp})
    fig.show()
    fig.savefig(f'{qfile_base}.png')


def q_31(f, T, fs):
    (cpx_cos, N) = cosine(fs, T, f)
    assert N == cpx_cos.size
    assert np.isreal(cpx_cos).all()

if __name__ == '__main__':
    q_31(1000,1,44800)

def q_32():
    f = 440
    T = 2
    fs = 44100
    (cpx_exp, N) = cosine(fs, T, f)
    return (cpx_exp, N, fs)
if __name__ == '__main__':
    soundpath = make_qdir('q_32')
    (cpx_cos, N, fs) = q_32()
    write_and_play(Path(soundpath, 'A440.wav'), fs, cpx_cos)
def q_33(fs):
    keys = np.arange(1,89)
    freqs = piano_key_to_frequency(keys)
    print()
    print(freqs)
    samples = np.arange(1,1)
    for f in freqs:
        cpx_cos = cosine(fs, 0.1, f)[0]
        samples = np.concatenate([samples, np.arange(0,1), cpx_cos])
    return samples

if __name__ == '__main__':
    fs = 192000
    samples = q_33(fs)
    write_and_play(Path(make_qdir('q_33'), '88keys.wav'), fs, samples)
def q_34(fs, file : Path):
    (keys, Ts, delays) = read_midi_to_keys(file)
    freqs = piano_key_to_frequency(np.asarray(keys))
    samples = list()
    for i in range(0, len(freqs)):
        cpx_cos = cosine(fs, Ts[i], freqs[i])[0]
        samples.append(np.zeros([int(fs*delays[i])]))
        samples.append(cpx_cos)
    return np.concatenate(samples)

if __name__ == '__main__':
    fs=44100
    q_34_dir = make_qdir('q_34')
    write_and_play(Path(q_34_dir, 'happybirthday.wav'), fs, q_34(fs, Path("HappyBirthday.mid")))
    write_and_play(Path(q_34_dir, 'NyanCat.wav'), fs, q_34(fs, Path("NyanCat.mid")))

if __name__ == '__main__':
    sys.exit(0)