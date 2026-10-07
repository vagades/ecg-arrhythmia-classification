"""
Извлечение вейвлет-признаков из ЭКГ-ударов.

Метод основан на дискретном вейвлет-преобразовании (ДВП) с вейвлетом Добеши
db4 и 4-уровневым разложением.

Теоретическая основа:
    Захарова Т.В., Шестаков О.В. «Теория вейвлетов и её применение в обработке
    сигналов». — М.: МастерПринт, 2018.

Частотные полосы для fs=360 Гц при 4-уровневом разложении:
    cD1 : 90–180 Гц  — высокочастотный шум
    cD2 : 45–90 Гц   — высокочастотные компоненты QRS
    cD3 : 22.5–45 Гц — комплекс QRS
    cD4 : 11.25–22.5 Гц — зубцы P и T
    cA4 : 0–11.25 Гц  — изолиния, медленные тренды
"""

import numpy as np
import pywt
from scipy.stats import skew


WAVELET = "db4"
LEVEL = 4


def extract_subband_features(coeffs: np.ndarray) -> np.ndarray:
    """5 статистических признаков из одной подполосы коэффициентов."""
    return np.array([
        np.mean(coeffs),
        np.std(coeffs),
        np.mean(coeffs ** 2),       # энергия
        np.max(np.abs(coeffs)),     # максимальный модуль
        float(skew(coeffs)),        # асимметрия
    ])


def extract_wavelet_features(beat: np.ndarray) -> np.ndarray:
    """
    Извлекает 25 вейвлет-признаков из одного ЭКГ-удара.

    Параметры
    ---------
    beat : ndarray, shape (240,)
        Нормализованный вектор одного кардиоцикла.

    Возвращает
    ----------
    features : ndarray, shape (25,)
        [cD1_stats | cD2_stats | cD3_stats | cD4_stats | cA4_stats],
        по 5 признаков на подполосу:
        (mean, std, energy, max_abs, skewness).
    """
    coeffs = pywt.wavedec(beat, WAVELET, level=LEVEL)
    # coeffs = [cA4, cD4, cD3, cD2, cD1]  — порядок pywt
    cA4, cD4, cD3, cD2, cD1 = coeffs

    features = np.concatenate([
        extract_subband_features(cD1),
        extract_subband_features(cD2),
        extract_subband_features(cD3),
        extract_subband_features(cD4),
        extract_subband_features(cA4),
    ])
    return features.astype(np.float32)


def build_wavelet_matrix(X: np.ndarray) -> np.ndarray:
    """
    Применяет extract_wavelet_features к каждой строке матрицы X.

    Параметры
    ---------
    X : ndarray, shape (N, 240)

    Возвращает
    ----------
    X_wt : ndarray, shape (N, 25)
    """
    return np.vstack([extract_wavelet_features(beat) for beat in X])


def build_combined_matrix(X: np.ndarray) -> np.ndarray:
    """
    Объединяет исходные 240 временных точек с 25 вейвлет-признаками.

    Возвращает ndarray shape (N, 265).
    """
    X_wt = build_wavelet_matrix(X)
    return np.hstack([X, X_wt])
