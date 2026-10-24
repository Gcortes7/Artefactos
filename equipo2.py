# -*- coding: utf-8 -*-
"""Interfaz del Equipo 2: procesamiento EEG mediante Wavelet-ICA."""

import os
import ema_core as ec


def ejecutar(csv, fs=None, channels=None, unit_scale=1.0,
             carpeta_salida="resultados_equipo2"):
    """Procesa un CSV EEG, guarda la señal limpia y genera una gráfica.

    Args:
        csv: ruta al CSV de entrada.
        fs: frecuencia de muestreo en Hz; si es None, ema_core intenta deducirla
            y, si no puede, usa 128 Hz.
        channels: lista opcional de nombres de canales EEG.
        unit_scale: factor de escala de amplitud, normalmente 1.0.
        carpeta_salida: carpeta donde se guardarán los resultados.
    """
    if not os.path.isfile(csv):
        raise FileNotFoundError(f"No se encontró el archivo CSV: {csv}")

    os.makedirs(carpeta_salida, exist_ok=True)
    nombre = os.path.splitext(os.path.basename(csv))[0]
    salida_csv = os.path.join(
        carpeta_salida, f"{nombre}_WaveletICA_Clean.csv"
    )

    print("\nIniciando procesamiento Wavelet-ICA...")
    r = ec.run_pipeline(
        csv,
        fs=fs,
        channels=channels,
        unit_scale=unit_scale
    )

    ec.save_clean_csv(r["data"], r["X_clean"], salida_csv,
                      unit_scale=unit_scale)

    eventos = r["events"]
    if len(eventos):
        canal = eventos.groupby("canal")["pico_a_pico_uV"].max().idxmax()
    else:
        canal = r["data"].names[0]

    prefijo_grafica = os.path.join(carpeta_salida, nombre)
    ec.plot_comparison(
        r["X"],
        r["X_clean"],
        r["data"].fs,
        r["data"].names,
        r["mask"],
        canal,
        prefijo_grafica
    )

    print(f"Eventos EMA detectados: {len(eventos)}")
    print(f"Componentes ICA tratadas: {r['info']['selected']}")
    print("\nMétricas por canal:")
    print(r["metrics"].to_string(index=False))
    print(f"\nCSV limpio guardado en: {salida_csv}")
    print(f"Gráficas guardadas con prefijo: {prefijo_grafica}")
    return r
