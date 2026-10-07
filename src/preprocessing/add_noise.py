import numpy as np


def add_gaussian_noise(signal, sigma=0.05, random_state=None):
    rng = np.random.default_rng(random_state)
    noise = rng.normal(0.0, sigma, size=signal.shape)
    return signal + noise


def add_baseline_wander(signal, amplitude=0.1, frequency=0.5, fs=360):
    t = np.arange(len(signal)) / fs
    drift = amplitude * np.sin(2 * np.pi * frequency * t)
    return signal + drift


def add_powerline_noise(signal, amplitude=0.05, frequency=50.0, fs=360):
    t = np.arange(len(signal)) / fs
    noise = amplitude * np.sin(2 * np.pi * frequency * t)
    return signal + noise


def add_muscle_noise(signal, amplitude=0.03, random_state=None):
    rng = np.random.default_rng(random_state)
    white = rng.normal(0.0, amplitude, size=signal.shape)

    kernel = np.array([1, -1, 1, -1, 1], dtype=float)
    kernel = kernel / np.sum(np.abs(kernel))
    emg_like = np.convolve(white, kernel, mode="same")

    return signal + emg_like