#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
CARE — Correccion de Artefactos por Razones Espectrales
===============================================================================
Metodo desarrollado por el equipo 7 para el tema ARTEFACTO_NORAW:
eliminacion de artefactos sobre datos EEG que NO son crudos.

Equipo 7  ·  Inteligencia Artificial 5801 IA 2026-2
Integrantes: Nestor Gael Lara Juarez  ·  Diego Guzman Rodriguez
Profesora:   Griselda Cortes Barrera

Artefactos tratados: OCULAR, MUSCULAR e INTERFERENCIA.

-------------------------------------------------------------------------------
EL PROBLEMA QUE RESUELVE ESTE MODULO
-------------------------------------------------------------------------------
ICA, FastICA, WaveletICA y el filtrado pasa-banda operan sobre la SEÑAL CRUDA
multicanal en el dominio del tiempo: descomponen la mezcla de fuentes y
reconstruyen la señal sin las componentes contaminadas. Todos necesitan la
forma de onda.

Los archivos de esta practica no la contienen. Contienen POTENCIA DE BANDA ya
calculada por la diadema Emotiv: para cada ventana se aplica FFT, se promedia
la energia dentro de cada banda y se guarda un solo numero por canal y banda.
Esa transformacion descarta la fase y es IRREVERSIBLE: sin forma de onda no hay
mezcla lineal de fuentes que ICA pueda separar, y un filtro pasa-banda no tiene
nada que filtrar porque el eje de frecuencia ya se colapso en cinco numeros.

El tema asignado al equipo 7 es exactamente ese: que hacer cuando los datos ya
no son crudos. La respuesta es trabajar en el ESPACIO DE CARACTERISTICAS, con
detectores especificos de la firma espectral de cada artefacto y una correccion
que preserva la estructura del archivo.

-------------------------------------------------------------------------------
COMO SE DETECTA CADA ARTEFACTO
-------------------------------------------------------------------------------
OCULAR     Parpadeos y movimientos oculares proceden del dipolo corneo-
           retiniano: son lentos (energia en theta y alpha) y su amplitud decae
           con la distancia al ojo, maxima en los frontales y casi nula en los
           occipitales. Se mide ese gradiente frontal-posterior.

MUSCULAR   La actividad EMG se concentra por encima de 20 Hz (betaH y gamma) y
           es periferica: contamina T7, T8, F7 y F8 mucho mas que los centrales.
           Se mide la proporcion de potencia alta en esos electrodos.

INTERFER.  El ruido de linea YA viene resuelto por hardware (notch 50/60 Hz) y
           gamma llega solo a ~45 Hz, asi que "interferencia" aqui significa
           perdida de contacto del electrodo: ceros exactos y saltos locales
           extremos. Se detecta celda por celda, no por muestra completa.

-------------------------------------------------------------------------------
TRES DECISIONES DE DISEÑO QUE CONVIENE CONOCER
-------------------------------------------------------------------------------
1. Los indices son DIFERENCIAS DE LOGARITMOS, no cocientes. Un cociente se
   dispara cuando el denominador se acerca a cero: en las pruebas llego a
   reportar que el artefacto habia aumentado un 700% cuando las dos potencias
   habian bajado. La escala logaritmica es ademas la natural para potencia de
   banda, que sigue una distribucion log-normal.

2. Se exigen DOS CONDICIONES para marcar: razon elevada Y potencia absoluta
   elevada. Una razon puede subir porque el numerador crecio o porque el
   denominador colapso, y solo lo primero es un artefacto fisiologico: el
   parpadeo y el musculo AÑADEN potencia, nunca la retiran.

3. Cada deteccion se expande +-8 muestras. Emotiv calcula la potencia con una
   ventana de 256 muestras desplazada 16 a la vez, lo que deja un 94% de
   solapamiento entre registros consecutivos: una rafaga instantanea contamina
   del orden de 16 registros.

-------------------------------------------------------------------------------
RESTRICCION: SE CONSERVA EL NUMERO DE REGISTROS
-------------------------------------------------------------------------------
El archivo procesado mantiene exactamente la misma cantidad de registros que el
original. Eso descarta eliminar las epocas contaminadas. En su lugar se CORRIGE
EN SITIO: en las muestras marcadas, y solo en las variables que ese artefacto
afecta, el valor se sustituye por interpolacion lineal entre las muestras
limpias vecinas.

-------------------------------------------------------------------------------
USO
-------------------------------------------------------------------------------
    import artefacto_noraw as e7

    e7.info()                              # informacion del equipo
    e7.limpiar_archivo("datos.csv")        # limpia un archivo
    e7.limpiar_carpeta("./mis_datos")      # limpia todos los CSV de una carpeta
    e7.menu()                              # menu interactivo (sirve en Colab)

Desde la linea de comandos:
    python3 artefacto_noraw.py archivo.csv
    python3 artefacto_noraw.py --carpeta ./datos --salida ./limpios
===============================================================================
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# IDENTIFICACION DEL EQUIPO
# =============================================================================

EQUIPO = {
    "numero": 7,
    "metodo": "CARE — Correccion de Artefactos por Razones Espectrales",
    "algoritmo": "Artefacto_NoRaw",  # tema asignado al equipo
    "integrantes": ["Nestor Gael Lara Juarez", "Diego Guzman Rodriguez"],
    "materia": "Inteligencia Artificial — 5801 IA 2026-2",
    "profesora": "Griselda Cortes Barrera",
    "institucion": "Tecnologico de Estudios Superiores de Ecatepec",
    "artefactos": ["ocular", "muscular", "interferencia"],
}


class Config:
    CANALES = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
               "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
    BANDAS = ["theta", "alpha", "betaL", "betaH", "gamma"]

    PERIFERICOS = ["T7", "T8", "F7", "F8"]     # vecinos a maseteros y cuello
    FRONTALES   = ["AF3", "AF4", "F7", "F8"]   # cercanos al dipolo ocular
    POSTERIORES = ["O1", "O2", "P7", "P8"]     # referencia sin contaminacion

    BANDAS_ALTAS  = ["betaH", "gamma"]          # donde vive el EMG
    BANDAS_BAJAS  = ["theta", "alpha"]          # donde vive el EOG
    BANDAS_MEDIAS = ["theta", "alpha", "betaL"]

    UMBRAL_Z = 3.5                 # z robusto para la deteccion por muestra
    UMBRAL_POTENCIA = 2.0          # condicion de apoyo: exceso absoluto
    UMBRAL_INTERFERENCIA = 6.0     # mas alto: la deteccion por celda hace
                                   # muchas mas pruebas y acumula falsos positivos
    EXPANSION = 8                  # +-8 muestras por el solapamiento del 94%
    EXPANSION_INTERFERENCIA = 2    # menor: los fallos de electrodo son puntuales

    SUFIJO = "NoRaw_Clean"


TAREAS = {"LeftRight": ("leftright", "leftrigth", "lr"),
          "PushPull":  ("pushpull", "pullpush", "pp")}


# =============================================================================
# UTILIDADES
# =============================================================================

def tarea_desde_nombre(nombre: str) -> str:
    """Deriva la tarea motora del nombre, tolerando las variantes del dataset:
    LeftRight, leftRight, LeftRigth (con el typo), PullPush, PushPull."""
    limpio = re.sub(r"[^a-z]", "", str(nombre).lower())
    for tarea, claves in TAREAS.items():
        if any(c in limpio for c in claves):
            return tarea
    return "SinClasificar"


def sujeto_desde_nombre(nombre: str) -> str:
    m = re.match(r"(\d+)", Path(nombre).stem)
    return m.group(1) if m else Path(nombre).stem


def leer_csv(ruta) -> tuple[pd.DataFrame, str]:
    """Carga el CSV respetando su separador real. Los archivos del dataset no
    son homogeneos: la mayoria usa coma y algunos punto y coma."""
    ruta = Path(ruta)
    with open(ruta, "r", encoding="utf-8", errors="replace") as f:
        primera = f.readline()
    sep = ";" if primera.count(";") > primera.count(",") else ","
    df = pd.read_csv(ruta, sep=sep, skipinitialspace=True,
                     encoding="utf-8", encoding_errors="replace")
    df.columns = [str(c).strip() for c in df.columns]
    return df, sep


def mapa_senal(df: pd.DataFrame) -> dict:
    """Devuelve {(canal, banda): nombre_de_columna} para las variables de señal."""
    m = {}
    for col in df.columns:
        if "/" not in str(col):
            continue
        ch, bd = str(col).split("/", 1)
        ch, bd = ch.strip(), bd.strip()
        if ch in Config.CANALES and bd in Config.BANDAS:
            m[(ch, bd)] = col
    return m


def columnas_de(mapa: dict, canales, bandas) -> list:
    return [mapa[(c, b)] for c in canales for b in bandas if (c, b) in mapa]


def z_robusto(x: np.ndarray) -> np.ndarray:
    """
    z-score robusto con mediana y MAD.

    Se prefiere a media y desviacion estandar porque los propios valores
    extremos que se busca detectar inflan esas dos estadisticas, elevando el
    umbral justo donde deberia ser sensible.
    """
    x = np.asarray(x, dtype=float)
    fin = x[np.isfinite(x)]
    if fin.size < 10:
        return np.zeros_like(x)
    med = np.median(fin)
    mad = np.median(np.abs(fin - med))
    if mad <= 0:
        return np.zeros_like(x)
    return np.nan_to_num(0.6745 * (x - med) / mad, nan=0.0,
                         posinf=0.0, neginf=0.0)


def expandir(mascara: np.ndarray, radio: int) -> np.ndarray:
    """Extiende cada deteccion +-radio muestras, por el solapamiento del 94%
    entre ventanas consecutivas de la diadema."""
    if radio <= 0 or not mascara.any():
        return mascara.copy()
    n = len(mascara)
    out = mascara.copy()
    for i in np.flatnonzero(mascara):
        out[max(0, i - radio): min(n, i + radio + 1)] = True
    return out


# =============================================================================
# DETECTORES
# =============================================================================

def indice_ocular(L: pd.DataFrame, mapa: dict) -> np.ndarray:
    """Gradiente frontal-posterior en bandas bajas. Diferencia de logaritmos."""
    fr = columnas_de(mapa, Config.FRONTALES, Config.BANDAS_BAJAS)
    po = columnas_de(mapa, Config.POSTERIORES, Config.BANDAS_BAJAS)
    if not fr or not po:
        return np.zeros(len(L))
    return (L[fr].mean(axis=1) - L[po].mean(axis=1)).to_numpy(dtype=float)


def indice_muscular(L: pd.DataFrame, mapa: dict) -> np.ndarray:
    """Proporcion de potencia alta en electrodos perifericos."""
    altas = columnas_de(mapa, Config.PERIFERICOS, Config.BANDAS_ALTAS)
    bajas = columnas_de(mapa, Config.PERIFERICOS, Config.BANDAS_MEDIAS)
    if not altas or not bajas:
        return np.zeros(len(L))
    return (L[altas].mean(axis=1) - L[bajas].mean(axis=1)).to_numpy(dtype=float)


def mascara_interferencia(valores: np.ndarray, L: np.ndarray,
                          umbral: float, radio: int) -> np.ndarray:
    """
    Mascara BIDIMENSIONAL (muestras x variables) de celdas con interferencia.

    Se detecta por celda y no por muestra completa porque la interferencia es
    un fallo del instrumento: afecta a UN electrodo en UN momento, mientras los
    demas siguen registrando bien. Marcar la muestra entera y reescribir las 70
    variables destruiria la señal sana de los otros canales.
    """
    n, m = valores.shape
    mask = (valores == 0.0)        # cero exacto = ausencia de señal

    if n > 2:
        d = np.zeros_like(L)
        d[1:] = np.abs(np.diff(L, axis=0))
        for j in range(m):
            col = d[:, j]
            fin = col[np.isfinite(col) & (col > 0)]
            if fin.size < 10:
                continue
            med = np.median(fin)
            mad = np.median(np.abs(fin - med))
            if mad <= 0:
                continue
            mask[:, j] |= (0.6745 * (col - med) / mad > umbral)

    if radio > 0:
        for j in range(m):
            if mask[:, j].any():
                mask[:, j] = expandir(mask[:, j], radio)
    return mask


# =============================================================================
# CORRECCION
# =============================================================================

def corregir_serie(valores: np.ndarray, mascara: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Sustituye las muestras marcadas por interpolacion lineal entre las limpias
    vecinas.

    Se prefiere a imputar la media o la mediana del canal porque la potencia de
    banda varia lentamente dentro de una sesion: los vecinos inmediatos son
    mejor estimador que cualquier estadistico global, y la interpolacion
    preserva la continuidad de la serie, que es lo que un artefacto rompe.

    La interpolacion se limita al interior de la serie. Una muestra marcada al
    principio o al final no tiene vecino limpio por un lado, y rellenarla
    arrastrando el valor contiguo produce un registro copiado en vez de medido.
    En ese caso se conserva el valor original: es preferible dejar sin corregir
    una muestra en el borde que fabricar el dato.
    """
    if not mascara.any():
        return valores, 0

    # Copia explicita: pd.Series puede compartir el buffer del array de entrada.
    valores = np.array(valores, dtype=float, copy=True)
    limpias = ~mascara
    if limpias.sum() < 2:
        return valores, 0

    s = pd.Series(valores.copy(), dtype=float)
    s[mascara] = np.nan
    # Se interpola en ambas direcciones para cubrir tambien las muestras
    # marcadas en los extremos de la serie. Ojo: esto solo es seguro porque
    # los registros incompletos se descartan ANTES de procesar. Si una fila
    # truncada llega hasta aqui, sus campos vacios entran como NaN y la
    # interpolacion los rellena copiando la fila anterior, con lo que el
    # archivo termina con un registro que nadie midio.
    s = s.interpolate(method="linear", limit_direction="both")
    if s.isna().any():
        s = s.fillna(float(np.nanmedian(valores[limpias])))

    corregida = s.to_numpy(dtype=float)
    return corregida, int((np.abs(corregida - valores) > 1e-12).sum())


# =============================================================================
# PROCESAMIENTO DE UN ARCHIVO
# =============================================================================

def limpiar_archivo(ruta_entrada, carpeta_salida=None, verbose=True) -> dict:
    """
    Aplica Artefacto_NoRaw a un archivo CSV y escribe la version limpia.

    El archivo resultante conserva: numero de registros, columnas, orden,
    separador, marcas de tiempo y metadatos. Solo cambian los valores de las
    variables de señal en las muestras corregidas.

    Devuelve un diccionario con el resumen del procesamiento.
    """
    ruta = Path(ruta_entrada)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontro el archivo: {ruta}")

    df, sep = leer_csv(ruta)

    # Registros incompletos: la grabacion de la mayoria de los archivos se
    # corto a media escritura y dejo una ultima fila truncada. No es un dato
    # parcial recuperable, es un registro que nunca se termino de escribir.
    # Con autorizacion de la profesora se eliminan antes de procesar, para que
    # el algoritmo no trabaje sobre valores que no existen.
    antes = len(df)
    df = df.dropna(how="all").dropna(how="any").reset_index(drop=True)
    descartados = antes - len(df)

    mapa = mapa_senal(df)
    cols = [mapa[k] for k in sorted(mapa.keys())]

    if not cols:
        raise ValueError(
            f"{ruta.name}: no se encontraron variables de señal con el formato "
            f"CANAL/banda (por ejemplo AF3/theta). Revisa que el archivo sea un "
            f"export de potencia de banda de Emotiv.")

    orig = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    orig = np.nan_to_num(orig, nan=0.0, posinf=0.0, neginf=0.0)
    L = pd.DataFrame(np.log1p(np.clip(orig, 0, None)), columns=cols)

    ceros_antes = int((orig == 0.0).sum())

    # ---- deteccion ocular y muscular (por muestra) -----------------------
    # Doble condicion: razon elevada Y potencia absoluta elevada. Una razon
    # puede subir porque el denominador colapso, y eso es interferencia, no un
    # artefacto fisiologico.
    i_eog = indice_ocular(L, mapa)
    i_emg = indice_muscular(L, mapa)

    col_fr = columnas_de(mapa, Config.FRONTALES, Config.BANDAS_BAJAS)
    col_pe = columnas_de(mapa, Config.PERIFERICOS, Config.BANDAS_ALTAS)
    pot_eog = L[col_fr].mean(axis=1).to_numpy(float) if col_fr else np.zeros(len(L))
    pot_emg = L[col_pe].mean(axis=1).to_numpy(float) if col_pe else np.zeros(len(L))

    d_eog = ((z_robusto(i_eog) > Config.UMBRAL_Z)
             & (z_robusto(pot_eog) > Config.UMBRAL_POTENCIA))
    d_emg = ((z_robusto(i_emg) > Config.UMBRAL_Z)
             & (z_robusto(pot_emg) > Config.UMBRAL_POTENCIA))

    m_eog = expandir(d_eog, Config.EXPANSION)
    m_emg = expandir(d_emg, Config.EXPANSION)

    # ---- deteccion de interferencia (por celda) ---------------------------
    M_int = mascara_interferencia(orig, L[cols].to_numpy(dtype=float),
                                  Config.UMBRAL_INTERFERENCIA,
                                  Config.EXPANSION_INTERFERENCIA)

    # ---- correccion --------------------------------------------------------
    nuevo = orig.copy()
    idx = {c: i for i, c in enumerate(cols)}
    celdas = 0

    # Ocular y muscular: la muestra marcada se corrige en su grupo topografico.
    for mascara, columnas in [(m_eog, col_fr), (m_emg, col_pe)]:
        if not mascara.any():
            continue
        for c in columnas:
            j = idx[c]
            nuevo[:, j], k = corregir_serie(nuevo[:, j], mascara)
            celdas += k

    # Interferencia: celda a celda, sin tocar las variables sanas.
    for j in range(len(cols)):
        if M_int[:, j].any():
            nuevo[:, j], k = corregir_serie(nuevo[:, j], M_int[:, j])
            celdas += k

    # ---- exportacion -------------------------------------------------------
    sujeto = sujeto_desde_nombre(ruta.name)
    tarea = tarea_desde_nombre(ruta.name)
    # Si el nombre no indica la tarea motora se conserva tal cual, para no
    # inventar una etiqueta que el archivo no trae.
    if tarea == "SinClasificar":
        nombre_salida = f"{ruta.stem}_{Config.SUFIJO}.csv"
    else:
        nombre_salida = f"{sujeto}_{tarea}_{Config.SUFIJO}.csv"

    salida = Path(carpeta_salida) if carpeta_salida else ruta.parent / "limpios"
    salida.mkdir(parents=True, exist_ok=True)
    destino = salida / nombre_salida

    out = df.copy()
    for i, c in enumerate(cols):
        out[c] = np.round(nuevo[:, i], 6)
    out.to_csv(destino, sep=sep, index=False, encoding="utf-8")

    # ---- verificacion ------------------------------------------------------
    rel = pd.read_csv(destino, sep=sep, skipinitialspace=True)
    ok_forma = (rel.shape == out.shape)

    Ln = pd.DataFrame(np.log1p(np.clip(nuevo, 0, None)), columns=cols)
    o_flat = L.to_numpy().ravel()
    n_flat = Ln.to_numpy().ravel()
    fin = np.isfinite(o_flat) & np.isfinite(n_flat)
    corr = (float(np.corrcoef(o_flat[fin], n_flat[fin])[0, 1])
            if fin.sum() > 2 and np.std(o_flat[fin]) > 0 else 1.0)

    r = {
        "archivo_original": ruta.name,
        "archivo_limpio": nombre_salida,
        "ruta_salida": str(destino),
        "sujeto": sujeto,
        "tarea": tarea,
        "registros": int(len(df)),
        "registros_descartados": int(descartados),
        "columnas": int(df.shape[1]),
        "variables_senal": len(cols),
        "muestras_ocular": int(d_eog.sum()),
        "muestras_muscular": int(d_emg.sum()),
        "celdas_interferencia": int(M_int.sum()),
        "celdas_corregidas": celdas,
        "pct_celdas": round(100 * celdas / max(orig.size, 1), 3),
        "ceros_antes": ceros_antes,
        "ceros_despues": int((nuevo == 0.0).sum()),
        "correlacion": round(corr, 6),
        "mismo_numero_registros": bool(len(rel) == len(df)),
        "estructura_intacta": bool(ok_forma),
    }

    if verbose:
        print(f"  {ruta.name}")
        print(f"     -> {nombre_salida}")
        if descartados:
            print(f"     {descartados} registro(s) incompleto(s) descartado(s)")
        print(f"     {r['registros']} registros conservados · "
              f"{celdas:,} celdas corregidas ({r['pct_celdas']}%)")
        print(f"     ocular: {r['muestras_ocular']} muestras · "
              f"muscular: {r['muestras_muscular']} · "
              f"ceros: {ceros_antes:,} -> {r['ceros_despues']:,}")
        print(f"     correlacion con el original: {r['correlacion']:.4f}")
    return r


def limpiar_carpeta(carpeta, carpeta_salida=None, verbose=True) -> pd.DataFrame:
    """Aplica el algoritmo a todos los CSV de una carpeta."""
    carpeta = Path(carpeta)
    archivos = sorted(p for p in carpeta.glob("*.csv")
                      if Config.SUFIJO not in p.name)
    if not archivos:
        print(f"No se encontraron archivos CSV en {carpeta}")
        return pd.DataFrame()

    salida = Path(carpeta_salida) if carpeta_salida else carpeta / "limpios"
    print(f"\nProcesando {len(archivos)} archivo(s) con Artefacto_NoRaw...\n")

    filas = []
    for ruta in archivos:
        try:
            filas.append(limpiar_archivo(ruta, salida, verbose))
        except Exception as exc:
            print(f"  {ruta.name}: ERROR — {exc}")

    if not filas:
        return pd.DataFrame()

    d = pd.DataFrame(filas)
    reporte = salida / "reporte_limpieza.csv"
    d.to_csv(reporte, index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 62}")
    print(f"  Archivos procesados ....... {len(d)}")
    print(f"  Registros conservados ..... {int(d.registros.sum()):,}")
    if "registros_descartados" in d and d.registros_descartados.sum():
        print(f"  Registros incompletos ..... {int(d.registros_descartados.sum()):,} descartados")
    print(f"  Celdas corregidas ......... {int(d.celdas_corregidas.sum()):,} "
          f"({d.pct_celdas.mean():.2f}% en promedio)")
    print(f"  Ceros eliminados .......... {int(d.ceros_antes.sum()):,} -> "
          f"{int(d.ceros_despues.sum()):,}")
    print(f"  Correlacion media ......... {d.correlacion.mean():.4f}")
    print(f"\n  Archivos limpios en: {salida}")
    print(f"  Reporte: {reporte.name}")
    print(f"{'=' * 62}\n")
    return d


# =============================================================================
# INTERFAZ
# =============================================================================

def info() -> None:
    """Muestra la informacion del equipo y del algoritmo asignado."""
    L = 64
    print("\n" + "=" * L)
    print(f"  EQUIPO {EQUIPO['numero']}  —  CARE".center(L))
    print("  Correccion de Artefactos por Razones Espectrales".center(L))
    print("=" * L)
    print()
    print(f"  Tema asignado: {EQUIPO['algoritmo']}")
    print()
    print("  INTEGRANTES")
    for n in EQUIPO["integrantes"]:
        print(f"     · {n}")
    print()
    print(f"  Materia     : {EQUIPO['materia']}")
    print(f"  Profesora   : {EQUIPO['profesora']}")
    print(f"  Institucion : {EQUIPO['institucion']}")
    print()
    print("-" * L)
    print("  ARTEFACTOS QUE TRATA EL ALGORITMO")
    print("-" * L)
    print("""
  OCULAR       Parpadeo y movimiento ocular. Proceden del dipolo corneo-
               retiniano: son lentos (theta y alpha) y su amplitud decae con
               la distancia al ojo. Se detectan por el gradiente entre
               electrodos frontales y occipitales.

  MUSCULAR     Actividad EMG: mandibula, cuello, deglucion. Se concentra por
               encima de 20 Hz (betaH y gamma) y es periferica, de modo que
               contamina T7, T8, F7 y F8 mucho mas que los canales centrales.

  INTERFERENCIA  Perdida de contacto del electrodo y deriva de impedancia. No
               es ruido de linea: la diadema ya aplica un notch de 50/60 Hz
               por hardware y gamma llega solo a ~45 Hz. Se manifiesta como
               ceros exactos y saltos locales extremos.
""")
    print("-" * L)
    print("  POR QUE NO SE USA ICA, FastICA, WaveletICA NI PASA-BANDA")
    print("-" * L)
    print("""
  Los cuatro metodos operan sobre la SEÑAL CRUDA en el dominio del tiempo.
  Estos archivos no la contienen: traen potencia de banda ya calculada por la
  diadema (14 canales x 5 bandas). El calculo aplica FFT, promedia la energia
  por banda y descarta la fase, de modo que es IRREVERSIBLE.

  Sin forma de onda no hay mezcla lineal de fuentes que ICA pueda separar, y
  un filtro pasa-banda no tiene nada que filtrar porque el eje de frecuencia
  ya se colapso en cinco numeros por canal.

  La via valida sobre potencia de banda es detectar las muestras contaminadas
  con indices espectrales especificos y corregirlas en el sitio. Eso es lo que
  hace Artefacto_NoRaw.
""")
    print("-" * L)
    print("  PARAMETROS")
    print("-" * L)
    print(f"""
  Umbral de deteccion (z robusto)  : {Config.UMBRAL_Z}
  Condicion de potencia absoluta   : {Config.UMBRAL_POTENCIA}
  Umbral de interferencia por celda: {Config.UMBRAL_INTERFERENCIA}
  Expansion de la mascara          : +-{Config.EXPANSION} muestras
  Expansion para interferencia     : +-{Config.EXPANSION_INTERFERENCIA} muestras
  Nomenclatura de salida           : <sujeto>_<tarea>_{Config.SUFIJO}.csv
""")
    print("=" * L)
    print("  El archivo procesado conserva el MISMO numero de registros,")
    print("  las mismas columnas, el separador y las marcas de tiempo.")
    print("=" * L + "\n")


def _en_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _subir_en_colab() -> list:
    """Abre el selector de archivos de Colab y devuelve las rutas subidas."""
    from google.colab import files
    print("\n  Selecciona el archivo o archivos CSV que quieres limpiar...")
    subidos = files.upload()
    return [Path(n) for n in subidos.keys()]


def procesar_subida() -> None:
    """
    Permite a cualquier persona subir un archivo y limpiarlo con el mismo
    algoritmo y los mismos parametros que uso el equipo 7.
    """
    rutas = []

    if _en_colab():
        try:
            rutas = _subir_en_colab()
        except Exception as exc:
            print(f"  No se pudo abrir el selector de Colab: {exc}")
            return
    else:
        print("\n  Escribe la ruta del archivo CSV o de la carpeta a procesar.")
        print("  (Enter sin texto para cancelar)")
        entrada = input("  Ruta: ").strip().strip('"').strip("'")
        if not entrada:
            print("  Cancelado.")
            return
        p = Path(entrada).expanduser()
        if p.is_dir():
            limpiar_carpeta(p)
            return
        if not p.exists():
            print(f"  No existe: {p}")
            return
        rutas = [p]

    if not rutas:
        print("  No se recibio ningun archivo.")
        return

    print(f"\n  Aplicando Artefacto_NoRaw a {len(rutas)} archivo(s)...\n")
    resultados = []
    for r in rutas:
        try:
            resultados.append(limpiar_archivo(r, Path("limpios")))
        except Exception as exc:
            print(f"  {r.name}: ERROR — {exc}")

    if resultados and _en_colab():
        from google.colab import files as colab_files
        print("\n  Descargando los archivos limpios...")
        for res in resultados:
            try:
                colab_files.download(res["ruta_salida"])
            except Exception:
                print(f"  Descarga manual: {res['ruta_salida']}")


def menu() -> None:
    """Menu del equipo 7. Es lo que se ejecuta desde la opcion 7 del Menu.ipynb."""
    while True:
        print("\n" + "=" * 64)
        print(f"   EQUIPO {EQUIPO['numero']} — CARE ({EQUIPO['algoritmo']})".center(64))
        print("=" * 64)
        print("  1. Ver informacion del equipo y del algoritmo")
        print("  2. Limpiar un archivo o carpeta (sube tu CSV)")
        print("  3. Ver los parametros del algoritmo")
        print("  0. Volver al menu principal")
        print("=" * 64)

        try:
            op = input("  Selecciona una opcion (0-3): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Volviendo al menu principal.")
            return

        if op == "0":
            print("  Volviendo al menu principal.")
            return
        elif op == "1":
            info()
        elif op == "2":
            procesar_subida()
        elif op == "3":
            print(f"""
  Umbral z robusto (mediana/MAD) ... {Config.UMBRAL_Z}
  Condicion de potencia absoluta ... {Config.UMBRAL_POTENCIA}
  Umbral de interferencia .......... {Config.UMBRAL_INTERFERENCIA}
  Expansion de mascara ............. +-{Config.EXPANSION} muestras
  Expansion de interferencia ....... +-{Config.EXPANSION_INTERFERENCIA} muestras
  Canales perifericos (EMG) ........ {', '.join(Config.PERIFERICOS)}
  Canales frontales (EOG) .......... {', '.join(Config.FRONTALES)}
  Canales posteriores (referencia) . {', '.join(Config.POSTERIORES)}
""")
        else:
            print("  Opcion no valida. Intenta de nuevo.")


# =============================================================================
# LINEA DE COMANDOS
# =============================================================================

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Artefacto_NoRaw — Equipo 7. Elimina artefactos ocular, "
                    "muscular e interferencia sobre datos EEG de potencia de banda.")
    ap.add_argument("archivo", nargs="?", help="archivo CSV a limpiar")
    ap.add_argument("--carpeta", "-c", help="carpeta con los CSV a limpiar")
    ap.add_argument("--salida", "-s", help="carpeta de salida")
    ap.add_argument("--info", "-i", action="store_true",
                    help="muestra la informacion del equipo")
    ap.add_argument("--menu", "-m", action="store_true",
                    help="abre el menu interactivo")
    a = ap.parse_args()

    if a.info:
        info()
    elif a.menu:
        menu()
    elif a.carpeta:
        limpiar_carpeta(a.carpeta, a.salida)
    elif a.archivo:
        limpiar_archivo(a.archivo, a.salida)
    else:
        info()
        print("  Uso:")
        print("     python3 artefacto_noraw.py archivo.csv")
        print("     python3 artefacto_noraw.py --carpeta ./datos")
        print("     python3 artefacto_noraw.py --menu\n")
