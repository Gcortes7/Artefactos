# -*- coding: utf-8 -*-
"""
ema_core.py  -  Núcleo del proyecto: Artefactos por Movimiento Ocular (EMA) en EEG
=============================================================================
Contiene TODAS las funciones que usan los tres scripts de la actividad:

    01_detectar_ema.py        -> detección automática de EMA + gráficas + reporte
    02_eliminar_waveletica.py -> eliminación con Wavelet-ICA (wICA) + comparación
    03_generar_clean_csv.py   -> exporta *_Clean.csv, valida y compara (sección 7)

Dependencias:  numpy, scipy, pandas, matplotlib, scikit-learn, PyWavelets
    pip install numpy scipy pandas matplotlib scikit-learn PyWavelets

CONVENCIONES
  * Los datos se manejan como matriz X de forma (n_muestras, n_canales).
  * Se asume que la amplitud está en microvoltios (µV). Si tu CSV está en
    otra unidad, ajusta el parámetro `unit_scale` (p. ej. 1e6 si viene en V).
"""
import os
import re
import json
import warnings
import numpy as np
import pandas as pd
import pywt
import matplotlib
matplotlib.use("Agg")                       # sin ventana: guarda PNG
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA

_trapezoid = getattr(np, "trapezoid", None) or np.trapz   # compatible con NumPy 1.x y 2.x

# --------------------------------------------------------------------------
# 0. PARÁMETROS POR DEFECTO (ajústalos según tus datasets)
# --------------------------------------------------------------------------
DET_DEFAULT = dict(
    band=(0.2, 8.0),   # banda (Hz) donde vive el movimiento ocular
    z_thr=4.0,         # umbral robusto (desviaciones MAD) para marcar evento
    min_amp=0.0,       # amplitud mínima (µV) del pico para aceptar el evento
    min_dur=0.08,      # duración mínima de un evento (s)
    max_gap=0.15,      # une eventos separados por menos de esto (s)
    pad=0.25,          # margen añadido a cada evento en la máscara (s)
)
WICA_DEFAULT = dict(
    wavelet="sym4",    # wavelet madre (sym4 / db4 se parecen a un parpadeo)
    k_thr=1.0,         # factor sobre el umbral universal de cada nivel
    max_ics=3,         # máximo de componentes oculares a tratar
    overlap_thr=1.8,   # cuánto más energía tiene la IC dentro del evento
    kurt_thr=5.0,      # curtosis mínima (componentes "puntiagudas")
    hp_fit=1.0,        # filtro pasa-altas (Hz) SOLO para ajustar el ICA
    random_state=0,
)

_NON_EEG_NAMES = {"time", "timestamp", "tiempo", "t", "index", "idx", "id",
                  "label", "labels", "class", "clase", "marker", "markers",
                  "event", "trigger", "target", "sample", "counter", "trial",
                  "epoch"}


# --------------------------------------------------------------------------
# 1. LECTURA / ESCRITURA DE CSV
# --------------------------------------------------------------------------
class EEGData:
    """Contenedor: DataFrame original + metadatos para reescribir igual."""
    def __init__(self, df, channels, fs, sep, has_header, time_col, path):
        self.df, self.channels, self.fs = df, channels, fs
        self.sep, self.has_header, self.time_col, self.path = sep, has_header, time_col, path
        self.names = [str(c) for c in channels]

    def matrix(self, unit_scale=1.0):
        """Devuelve X (n, c) en float64, interpolando NaN si los hubiera."""
        sub = self.df[self.channels].apply(pd.to_numeric, errors="coerce")
        if sub.isna().any().any():
            warnings.warn("Se encontraron NaN en canales; se interpolan para procesar.")
            sub = sub.interpolate(limit_direction="both")
        return sub.to_numpy(dtype=float) * unit_scale


def load_eeg_csv(path, fs=None, channels=None, unit_scale=1.0):
    """
    Lee un CSV de EEG. Detecta separador, encabezado, columna de tiempo y
    canales (columnas numéricas que no sean tiempo/etiqueta/marcador).
    - channels: lista de nombres (o índices) para forzar los canales.
    - fs: frecuencia de muestreo en Hz; si es None se intenta deducir de la
      columna de tiempo y, si no existe, se usa 128 Hz con un aviso.
    """
    df, sep = None, ","
    for s in (",", ";", "\t"):
        tmp = pd.read_csv(path, sep=s)
        if tmp.shape[1] > 1:
            df, sep = tmp, s
            break
    if df is None:
        df = pd.read_csv(path, sep=None, engine="python")
    # ¿sin encabezado? (todos los nombres de columna parecen números)
    has_header = True
    try:
        [float(c) for c in df.columns]
        df = pd.read_csv(path, sep=sep, header=None)
        has_header = False
    except (ValueError, TypeError):
        pass

    time_col = None
    for c in df.columns:
        n = str(c).lower()
        if n in ("time", "timestamp", "tiempo", "t") or "time" in n:
            time_col = c
            break

    if channels:
        chs = [df.columns[int(c)] if str(c).isdigit() and not has_header else c
               for c in channels]
    else:
        chs = []
        for c in df.columns:
            n = str(c).lower().strip()
            if c == time_col or n in _NON_EEG_NAMES:
                continue
            if not pd.api.types.is_numeric_dtype(df[c]):
                continue
            v = df[c].dropna()
            if len(v) and np.allclose(v, np.round(v)) and v.nunique() <= 12:
                continue                      # parece etiqueta/clase
            chs.append(c)
    if not chs:
        raise ValueError("No se detectaron canales EEG; usa el argumento channels.")

    if fs is None:
        if time_col is not None:
            dt = np.median(np.diff(pd.to_numeric(df[time_col], errors="coerce").dropna()))
            if dt > 0:
                fs = 1.0 / dt if dt < 1 else 1000.0 / dt
                print(f"[info] fs deducida de '{time_col}': {fs:.2f} Hz")
        if fs is None:
            fs = 128.0
            warnings.warn("No se pudo deducir fs; se usa 128 Hz. Pásala con --fs.")
    return EEGData(df, chs, float(fs), sep, has_header, time_col, path)


def save_clean_csv(data, X_clean, out_path, unit_scale=1.0):
    """Escribe un CSV idéntico al original salvo por los canales EEG corregidos."""
    out = data.df.copy()
    out[data.channels] = X_clean / unit_scale
    out.to_csv(out_path, index=False, sep=data.sep, header=data.has_header)
    return out_path


# --------------------------------------------------------------------------
# 2. UTILIDADES DE SEÑAL
# --------------------------------------------------------------------------
def bandpass(X, fs, lo, hi, order=4):
    """Filtro Butterworth de fase cero (sosfiltfilt) sobre el eje 0."""
    nyq = fs / 2.0
    hi = min(hi, 0.99 * nyq)
    if lo <= 0:
        sos = butter(order, hi / nyq, btype="low", output="sos")
    else:
        sos = butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, X, axis=0)


def highpass(X, fs, fc, order=4):
    sos = butter(order, fc / (fs / 2.0), btype="high", output="sos")
    return sosfiltfilt(sos, X, axis=0)


def bandpower(x, fs, lo, hi):
    """Potencia en [lo, hi] Hz con el método de Welch (integral de la PSD)."""
    f, p = welch(x, fs=fs, nperseg=min(len(x), int(fs * 4)))
    m = (f >= lo) & (f <= hi)
    return float(_trapezoid(p[m], f[m])) if m.any() else 0.0


def is_frontal(name):
    """True para Fp1, Fp2, AF3, AF4, F3, F4, F7, F8, Fz (admite prefijos tipo 'EEG.F7')."""
    tok = re.split(r"[._\s-]", str(name).strip())[-1].upper()
    return bool(re.match(r"^(FP|AF|F)(Z|\d+)$", tok))


def _mask_to_segments(mask, fs, min_dur, max_gap):
    """Convierte una máscara booleana en segmentos (i0, i1) fusionando huecos cortos."""
    d = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    idx = np.flatnonzero(d)
    merged = []
    for s, e in zip(idx[0::2], idx[1::2]):
        if merged and (s - merged[-1][1]) <= max_gap * fs:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if (e - s) / fs >= min_dur]


def _dilate(mask, n):
    if n <= 0 or not mask.any():
        return mask.copy()
    k = np.ones(2 * n + 1)
    return np.convolve(mask.astype(float), k, mode="same") > 0


# --------------------------------------------------------------------------
# 3. DETECCIÓN AUTOMÁTICA DE EMA
# --------------------------------------------------------------------------
def detect_ema(X, fs, names, **kw):
    """
    Detecta movimientos oculares (sacadas, parpadeos, desplazamientos) en
    cada canal.

    Método (umbral robusto en la banda ocular):
      1) Filtro pasa-banda 0.5-8 Hz  -> aísla la banda del EOG (< 4 Hz)
         y descarta alfa/beta.
      2) z = |x - mediana| / (1.4826 * MAD)  -> amplitud normalizada
         resistente a valores atípicos (el propio artefacto no infla el umbral).
      3) Evento = z > z_thr durante al menos `min_dur` s (y pico >= min_amp).
      4) Se fusionan eventos separados por < `max_gap` s.

    Devuelve
      events : DataFrame (canal, inicio_s, fin_s, duracion_s, amp_pico_uV,
               pico_a_pico_uV, z_max, polaridad, tipo_estimado)
      mask   : bool (n,) unión de eventos (canales frontales si existen)
      Xl     : señal filtrada en banda ocular (para graficar)
    """
    p = {**DET_DEFAULT, **kw}
    n = X.shape[0]
    Xl = bandpass(X - np.median(X, axis=0), fs, *p["band"])
    med = np.median(Xl, axis=0)
    mad = np.median(np.abs(Xl - med), axis=0) * 1.4826 + 1e-12
    Z = np.abs(Xl - med) / mad

    rows, chmask = [], np.zeros(X.shape, dtype=bool)
    for j, name in enumerate(names):
        for s, e in _mask_to_segments(Z[:, j] > p["z_thr"], fs, p["min_dur"], p["max_gap"]):
            seg = Xl[s:e, j] - med[j]
            k = int(np.argmax(np.abs(seg)))
            if abs(seg[k]) < p["min_amp"]:
                continue
            dur = (e - s) / fs
            rows.append(dict(
                canal=name, inicio_s=round(s / fs, 3), fin_s=round(e / fs, 3),
                duracion_s=round(dur, 3), amp_pico_uV=round(float(seg[k]), 2),
                pico_a_pico_uV=round(float(seg.max() - seg.min()), 2),
                z_max=round(float(Z[s:e, j].max()), 2),
                polaridad="+" if seg[k] > 0 else "-",
                tipo_estimado=("parpadeo / pico breve" if dur < 0.45
                               else "sacada / mov. horizontal sostenido")))
            chmask[s:e, j] = True
    events = pd.DataFrame(rows, columns=[
        "canal", "inicio_s", "fin_s", "duracion_s", "amp_pico_uV",
        "pico_a_pico_uV", "z_max", "polaridad", "tipo_estimado"])

    front = [j for j, nm in enumerate(names) if is_frontal(nm)]
    cols = front if front else list(range(len(names)))
    mask = _dilate(chmask[:, cols].any(axis=1), int(p["pad"] * fs))
    return events, mask, Xl


# --------------------------------------------------------------------------
# 4. WAVELET-ICA (wICA)
# --------------------------------------------------------------------------
def score_components(S, A, fs, mask, names):
    """
    Puntúa cada componente independiente (IC) para decidir si es ocular:
      overlap : energía media |IC| dentro de eventos / fuera de eventos
      kurtosis: las sacadas/parpadeos producen componentes muy puntiagudas
      low_frac: fracción de potencia en 0.5-4 Hz respecto a 0.5-40 Hz
      frontal : peso de los electrodos frontales en el mapa de mezcla
    """
    fr = [j for j, nm in enumerate(names) if is_frontal(nm)]
    rows = []
    for i in range(S.shape[1]):
        s = S[:, i]
        z = (s - s.mean()) / (s.std() + 1e-12)
        ov = (np.mean(np.abs(z[mask])) / (np.mean(np.abs(z[~mask])) + 1e-12)
              if mask.any() and (~mask).any() else np.nan)
        w = np.abs(A[:, i])
        rows.append(dict(
            IC=i, overlap=ov, kurtosis=float(kurtosis(s)),
            low_frac=bandpower(s, fs, 0.5, 4) / (bandpower(s, fs, 0.5, 40) + 1e-12),
            frontal=float(w[fr].sum() / w.sum()) if fr else np.nan))
    return pd.DataFrame(rows)


def select_ocular_ics(scores, **kw):
    """Regla de selección automática (ver WICA_DEFAULT)."""
    p = {**WICA_DEFAULT, **kw}
    ok = (scores.overlap >= p["overlap_thr"]) & (
        (scores["low_frac"] >= 0.4) | (scores["kurtosis"] >= p["kurt_thr"]))
    sel = scores[ok].sort_values("overlap", ascending=False).IC.tolist()[:p["max_ics"]]
    if not sel and scores.overlap.notna().any():           # plan B: la mejor IC
        best = scores.sort_values("overlap", ascending=False).iloc[0]
        if best.overlap >= 1.3:
            sel = [int(best.IC)]
    return sel


def wavelet_clean_ic(s, fs, mask, wavelet="sym4", k_thr=1.0, level=None):
    """
    Limpia UNA componente con umbralización wavelet (idea de Castellanos &
    Makarov, 2006): los coeficientes muy grandes de cada nivel pertenecen al
    artefacto; se anulan y se reconstruye. Así se conserva la actividad
    cerebral residual que ICA dejó dentro de la componente.

    Umbral por nivel j:  T_j = k * sigma_j * sqrt(2 ln N_j)
        sigma_j = MAD(c_j) / 0.6745, estimada con coeficientes FUERA de
        los eventos oculares (para que el artefacto no infle el umbral).
    """
    n = len(s)
    if level is None:
        level = max(3, int(np.ceil(np.log2(fs))) - 1)      # cA llega a ~1 Hz
    level = min(level, pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len))
    coeffs = pywt.wavedec(s - np.median(s), wavelet, level=level, mode="symmetric")
    out = []
    for i, c in enumerate(coeffs):
        lev = level if i == 0 else level - i + 1
        step = 2 ** lev
        pos = np.minimum(np.arange(len(c)) * step + step // 2, n - 1)
        ref = c[~mask[pos]] if (~mask[pos]).sum() >= 20 else c
        sigma = np.median(np.abs(ref - np.median(ref))) / 0.6745 + 1e-12
        T = k_thr * sigma * np.sqrt(2 * np.log(max(len(c), 2)))
        out.append(np.where(np.abs(c - np.median(ref)) > T, 0.0, c))
    return pywt.waverec(out, wavelet, mode="symmetric")[:n] + np.median(s)


def wavelet_ica_clean(X, fs, names, mask, n_components=None, **kw):
    """
    Pipeline Wavelet-ICA completo:
      1) FastICA (ajustado con datos filtrados > hp_fit Hz) -> S = fuentes.
      2) Puntuar ICs y elegir las oculares (usando la máscara de EMA).
      3) Limpiar SOLO esas ICs con umbralización wavelet.
      4) Reconstruir: X_clean = X + [A·S_limpia - A·S_original]
         (solo cambia la parte atribuida al artefacto; el resto queda intacto).
    Devuelve X_clean, info(dict con scores y componentes elegidas).
    """
    p = {**WICA_DEFAULT, **kw}
    dc = np.median(X, axis=0)
    Xc = X - dc                                            # quita offset DC
    Xfit = highpass(Xc, fs, p["hp_fit"]) if p["hp_fit"] and fs > 4 * p["hp_fit"] else Xc
    ica = FastICA(n_components=n_components or X.shape[1], whiten="unit-variance",
                  max_iter=2000, tol=1e-4, random_state=p["random_state"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ica.fit(Xfit)
    S = ica.transform(Xc)
    scores = score_components(S, ica.mixing_, fs, mask, names)
    sel = select_ocular_ics(scores, **p)
    S2 = S.copy()
    for i in sel:
        S2[:, i] = wavelet_clean_ic(S[:, i], fs, mask, p["wavelet"], p["k_thr"])
    X_clean = X + (ica.inverse_transform(S2) - ica.inverse_transform(S))
    return X_clean, dict(scores=scores, selected=sel, sources=S, sources_clean=S2,
                         mixing=ica.mixing_, params=p)


# --------------------------------------------------------------------------
# 5. MÉTRICAS Y ESTADÍSTICAS
# --------------------------------------------------------------------------
def evaluation_metrics(X, Xc, fs, names, mask):
    """
    Métricas por canal (original vs. corregida):
      reduccion_delta_pct : caída de potencia 0.5-4 Hz (donde está el EOG)
      reduccion_rms_evt_pct: caída del RMS dentro de los segmentos con EMA
      corr_fuera_evt      : correlación fuera de eventos. OJO: en canales frontales puede
                            salir baja porque la máscara no cubre todo el artefacto (p. ej. la
                            meseta de una sacada) y ahí la corrección SÍ debe cambiar la señal;
                            los canales posteriores (T7, O1, P8...) sirven de control (≈1)
      rmse_fuera_evt_uV   : error cuadrático fuera de eventos
      ratio_alfa, ratio_beta: potencia_después / potencia_antes (≈1 = conserva)
      kurt_antes/despues  : curtosis (los picos oculares la elevan)
    """
    out_m = ~_dilate(mask, int(0.1 * fs))
    rows = []
    for j, nm in enumerate(names):
        a, b = X[:, j] - np.median(X[:, j]), Xc[:, j] - np.median(Xc[:, j])
        d0, d1 = bandpower(a, fs, 0.5, 4), bandpower(b, fs, 0.5, 4)
        r = lambda v: np.sqrt(np.mean(v ** 2)) + 1e-12
        rows.append(dict(
            canal=nm,
            reduccion_delta_pct=round(100 * (1 - d1 / (d0 + 1e-12)), 2),
            reduccion_rms_evt_pct=round(100 * (1 - r(b[mask]) / r(a[mask])), 2) if mask.any() else np.nan,
            corr_fuera_evt=round(float(np.corrcoef(a[out_m], b[out_m])[0, 1]), 4) if out_m.sum() > 10 else np.nan,
            rmse_fuera_evt_uV=round(float(np.sqrt(np.mean((a[out_m] - b[out_m]) ** 2))), 3) if out_m.any() else np.nan,
            ratio_alfa=round(bandpower(b, fs, 8, 13) / (bandpower(a, fs, 8, 13) + 1e-12), 3),
            ratio_beta=round(bandpower(b, fs, 13, 30) / (bandpower(a, fs, 13, 30) + 1e-12), 3),
            kurt_antes=round(float(kurtosis(a)), 2), kurt_despues=round(float(kurtosis(b)), 2)))
    return pd.DataFrame(rows)


def descriptive_stats(X, names):
    """Estadísticas descriptivas por canal (para la comparación 7.4)."""
    return pd.DataFrame({
        "canal": names, "n_registros": X.shape[0], "media": X.mean(0), "desv_std": X.std(0),
        "minimo": X.min(0), "maximo": X.max(0), "pico_a_pico": np.ptp(X, axis=0),
        "rms": np.sqrt(np.mean((X - X.mean(0)) ** 2, axis=0)),
        "curtosis": kurtosis(X, axis=0)}).round(3)


# --------------------------------------------------------------------------
# 6. GRÁFICAS
# --------------------------------------------------------------------------
def _top_channels(events, names, k=4):
    if len(events):
        order = events.groupby("canal").pico_a_pico_uV.max().sort_values(ascending=False).index.tolist()
        return order[:k]
    return names[:k]


def plot_detection(X, fs, names, events, out_prefix, k_channels=4, n_zoom=4):
    """Fig. A: señal original con eventos sombreados. Fig. B: zoom de eventos."""
    t = np.arange(X.shape[0]) / fs
    chs = _top_channels(events, names, k_channels)
    fig, axs = plt.subplots(len(chs), 1, figsize=(13, 2.4 * len(chs)), sharex=True)
    axs = np.atleast_1d(axs)
    for ax, c in zip(axs, chs):
        j = names.index(c)
        ax.plot(t, X[:, j] - np.median(X[:, j]), lw=0.6, color="#1f4e79")
        for _, ev in events[events.canal == c].iterrows():
            ax.axvspan(ev.inicio_s, ev.fin_s, color="red", alpha=0.25)
        ax.set_ylabel("Amplitud (µV)")
        ax.set_title(f"Canal {c}: EEG original con eventos EMA detectados (sombreado rojo)", fontsize=10)
        ax.grid(alpha=0.3)
    axs[-1].set_xlabel("Tiempo (s)")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_deteccion.png", dpi=130)
    plt.close(fig)

    if len(events):
        top = events.sort_values("pico_a_pico_uV", ascending=False).head(n_zoom)
        fig, axs = plt.subplots(len(top), 1, figsize=(11, 2.6 * len(top)))
        axs = np.atleast_1d(axs)
        for ax, (_, ev) in zip(axs, top.iterrows()):
            j = names.index(ev.canal)
            i0, i1 = max(int((ev.inicio_s - 1) * fs), 0), min(int((ev.fin_s + 1) * fs), X.shape[0])
            ax.plot(t[i0:i1], X[i0:i1, j] - np.median(X[:, j]), color="#1f4e79", lw=0.9)
            ax.axvspan(ev.inicio_s, ev.fin_s, color="red", alpha=0.25)
            ax.set_title(f"Zoom canal {ev.canal} | t={ev.inicio_s}s, dur={ev.duracion_s}s, "
                         f"pico a pico={ev.pico_a_pico_uV} µV | {ev.tipo_estimado}", fontsize=9)
            ax.set_xlabel("Tiempo (s)"); ax.set_ylabel("Amplitud (µV)"); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{out_prefix}_zoom_eventos.png", dpi=130)
        plt.close(fig)


def plot_comparison(X, Xc, fs, names, mask, ch, out_prefix):
    """Original / con artefacto marcado / corregida + PSD antes-después."""
    j = names.index(ch)
    t = np.arange(X.shape[0]) / fs
    a, b = X[:, j] - np.median(X[:, j]), Xc[:, j] - np.median(Xc[:, j])
    
    # Limpiamos el nombre del canal para evitar errores de rutas con '/' o '\'
    ch_clean = str(ch).strip().replace("/", "_").replace("\\", "_")

    fig, axs = plt.subplots(4, 1, figsize=(13, 9), sharex=False)
    axs[0].plot(t, a, lw=0.6, color="#1f4e79"); axs[0].set_title(f"Canal {ch}: señal original")
    axs[1].plot(t, a, lw=0.6, color="gray")
    axs[1].plot(t, np.where(mask, a, np.nan), lw=0.8, color="red")
    axs[1].set_title(f"Canal {ch}: señal con artefacto identificado (rojo)")
    axs[2].plot(t, b, lw=0.6, color="#2e7d32"); axs[2].set_title(f"Canal {ch}: señal corregida (Wavelet-ICA)")
    lim = max(np.abs(a).max(), 1)
    for ax in axs[:3]:
        ax.set_ylim(-1.1 * lim, 1.1 * lim); ax.set_ylabel("Amplitud (µV)"); ax.grid(alpha=0.3)
    axs[0].set_xlabel("")
    # zoom sobre el evento más grande
    if mask.any():
        seg = _mask_to_segments(mask, fs, 0, 0)
        s, e = max(seg, key=lambda z: np.ptp(a[z[0]:z[1]]))
        i0, i1 = max(s - int(fs), 0), min(e + int(fs), len(a))
        axs[3].plot(t[i0:i1], a[i0:i1], color="gray", lw=1.0, label="original")
        axs[3].plot(t[i0:i1], b[i0:i1], color="#2e7d32", lw=1.0, label="corregida")
        axs[3].legend(loc="upper right")
    axs[3].set_title(f"Canal {ch}: zoom del evento más intenso (superposición)")
    axs[3].set_xlabel("Tiempo (s)"); axs[3].set_ylabel("Amplitud (µV)"); axs[3].grid(alpha=0.3)
    
    # Se guarda usando ch_clean en lugar de ch
    fig.tight_layout(); fig.savefig(f"{out_prefix}_comparacion_{ch_clean}.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for sig, lab, col in ((a, "original", "#1f4e79"), (b, "corregida", "#2e7d32")):
        f, p = welch(sig, fs=fs, nperseg=min(len(sig), int(fs * 4)))
        ax.semilogy(f, p, label=lab, color=col)
    ax.set_xlim(0, min(45, fs / 2)); ax.set_xlabel("Frecuencia (Hz)"); ax.set_ylabel("PSD (µV²/Hz)")
    ax.set_title(f"Canal {ch}: espectro antes y después"); ax.legend(); ax.grid(alpha=0.3)
    
    # Se guarda usando ch_clean en lugar de ch
    fig.tight_layout(); fig.savefig(f"{out_prefix}_psd_{ch_clean}.png", dpi=130); plt.close(fig)

# --------------------------------------------------------------------------
# 7. PIPELINE COMPLETO Y VALIDACIÓN (sección 7 de la actividad)
# --------------------------------------------------------------------------
def run_pipeline(path, fs=None, channels=None, unit_scale=1.0, det=None, wica=None):
    """Lee CSV -> detecta EMA -> Wavelet-ICA -> métricas. Devuelve dict."""
    det, wica = det or {}, wica or {}
    data = load_eeg_csv(path, fs, channels, unit_scale)
    X = data.matrix(unit_scale)
    ev, mask, _ = detect_ema(X, data.fs, data.names, **det)
    Xc, info = wavelet_ica_clean(X, data.fs, data.names, mask, **wica)
    met = evaluation_metrics(X, Xc, data.fs, data.names, mask)
    return dict(data=data, X=X, X_clean=Xc, events=ev, mask=mask, info=info, metrics=met)


def validate_clean_csv(orig_path, clean_path, fs=None, channels=None, unit_scale=1.0, det=None):
    """
    Valida el archivo exportado (sección 7.3). Devuelve (DataFrame de pruebas, dict resumen).
    Pruebas: mismo nº de registros/columnas/nombres, columnas no-EEG idénticas,
    sin NaN/Inf nuevos, artefacto reducido (re-detección) y cerebro conservado.
    """
    det = det or {}
    o = load_eeg_csv(orig_path, fs, channels, unit_scale)
    c = load_eeg_csv(clean_path, o.fs, [str(x) for x in o.channels] if o.has_header else None, unit_scale)
    Xo, Xc = o.matrix(unit_scale), c.matrix(unit_scale)
    other = [x for x in o.df.columns if x not in o.channels]
    tests = []
    add = lambda n, ok, d="": tests.append(dict(prueba=n, resultado="OK" if ok else "FALLA", detalle=d))
    add("Misma cantidad de registros", len(o.df) == len(c.df), f"{len(o.df)} vs {len(c.df)}")
    add("Mismas columnas y orden", list(o.df.columns) == list(c.df.columns))
    add("Mismo nº de canales EEG", Xo.shape[1] == Xc.shape[1], str(Xo.shape[1]))
    add("Columnas no-EEG (etiquetas/tiempo/meta) idénticas",
        all(o.df[k].equals(c.df[k]) for k in other), ", ".join(map(str, other)) or "ninguna")
    add("Sin NaN/Inf nuevos en canales", np.isfinite(Xc).all())
    ev0, m0, _ = detect_ema(Xo, o.fs, o.names, **det)
    ev1, m1, _ = detect_ema(Xc, o.fs, o.names, **det)
    red = 100 * (1 - m1.mean() / m0.mean()) if m0.any() else np.nan
    add("Artefacto reducido (tiempo contaminado)", (not m0.any()) or m1.mean() < m0.mean(),
        f"{100*m0.mean():.2f}% -> {100*m1.mean():.2f}% del registro; eventos {len(ev0)} -> {len(ev1)}")
    met = evaluation_metrics(Xo, Xc, o.fs, o.names, m0)
    # Canales de control = los poco afectados por EMA (<= 20 % de eventos del canal más afectado).
    # En ellos la señal NO debería cambiar; en los frontales sí (ahí se quita el artefacto).
    cnt = ev0.canal.value_counts() if len(ev0) else pd.Series(dtype=int)
    mx = cnt.max() if len(cnt) else 0
    ctrl = met[[cnt.get(nm, 0) <= 0.2 * mx for nm in met.canal]]
    ctrl = ctrl if len(ctrl) else met
    fro = met[~met.canal.isin(ctrl.canal)]
    add("Actividad cerebral conservada en canales de control (corr. >= 0.95)",
        bool((ctrl.corr_fuera_evt.dropna() >= 0.95).all()) if m0.any() else True,
        f"{len(ctrl)} canales poco afectados; corr. mínima = {ctrl.corr_fuera_evt.min():.4f}")
    add("Potencia alfa/beta conservada en canales de control (0.9-1.1)",
        bool(ctrl.ratio_alfa.between(0.9, 1.1).all() and ctrl.ratio_beta.between(0.9, 1.1).all()),
        f"control: alfa {ctrl.ratio_alfa.min():.2f}-{ctrl.ratio_alfa.max():.2f}, beta {ctrl.ratio_beta.min():.2f}-{ctrl.ratio_beta.max():.2f}"
        + (f" | canales afectados (informativo): alfa {fro.ratio_alfa.min():.2f}-{fro.ratio_alfa.max():.2f}, "
           f"beta {fro.ratio_beta.min():.2f}-{fro.ratio_beta.max():.2f}" if len(fro) else ""))
    bp = {nm: {b: round(bandpower(Xc[:, k] - Xc[:, k].mean(), o.fs, lo, hi), 3)
               for b, (lo, hi) in dict(delta=(1, 4), theta=(4, 8), alfa=(8, 13), beta=(13, 30)).items()}
          for k, nm in enumerate(o.names)}
    add("Reutilizable en etapas posteriores (se recargan y calculan potencias por banda)", bool(bp))
    summary = dict(eventos_antes=int(len(ev0)), eventos_despues=int(len(ev1)),
                   pct_tiempo_contaminado_antes=round(100 * m0.mean(), 2),
                   pct_tiempo_contaminado_despues=round(100 * m1.mean(), 2),
                   reduccion_tiempo_contaminado_pct=None if np.isnan(red) else round(red, 1))
    return pd.DataFrame(tests), summary, met
