
import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning


# ============================================================
# 1. MENU PRINCIPAL
# ============================================================

def mostrar_menu():
    print("\n" + "=" * 50)
    print("       MENÚ DE PROCESAMIENTO DE SEÑALES")
    print("=" * 50)
    print("1. Equipo 1 - Fast_ICA")
    print("2. Equipo 2 - Wavelet_ICA")
    print("3. Equipo 3 - Pasaband")
    print("4. Equipo 4 - InterferenciaModifier")
    print("5. Equipo 5 - ICA")
    print("6. Equipo 6 - ExtendInfomax_ICA")
    print("7. Equipo 7 - Artefacto_NoRaw")
    print("0. Salir")
    print("=" * 50)


# ============================================================
# 2. LIMPIAR NOMBRES
# ============================================================

def limpiar_nombre(nombre):
    return re.sub(r'[<>:"/\\|?*]', "_", str(nombre)).strip()


# ============================================================
# 3. PROCESAMIENTO EEG CON FASTICA
# ============================================================

def cargar_y_procesar_eeg(ruta_csv, carpeta_salida):

    nombre_base = os.path.splitext(
        os.path.basename(ruta_csv)
    )[0]

    print(f"\n[INFO] Procesando archivo: {nombre_base}")

    # Leer archivo CSV
    df = pd.read_csv(ruta_csv)

    print(f"[INFO] Dimensiones originales: {df.shape}")

    # Seleccionar columnas numéricas y primeras 1000 muestras
    df_num = df.select_dtypes(
        include=[np.number]
    ).iloc[:1000].copy()

    if df_num.empty:
        raise ValueError(
            "El archivo no contiene columnas numéricas."
        )

    # Sustituir infinitos por valores ausentes
    df_num = df_num.replace(
        [np.inf, -np.inf], np.nan
    )

    # Eliminar columnas completamente vacías
    df_num = df_num.dropna(
        axis=1, how="all"
    )

    if df_num.empty or len(df_num) < 2:
        raise ValueError(
            "No hay suficientes datos numéricos."
        )

    nombres_columnas = list(df_num.columns)

    # Guardar datos originales en memoria para las gráficas
    X_original = df_num.to_numpy(dtype=float)

    # Rellenar datos faltantes con la media de cada canal
    medias = np.nanmean(X_original, axis=0)

    filas_nan, columnas_nan = np.where(
        np.isnan(X_original)
    )

    X = X_original.copy()

    X[filas_nan, columnas_nan] = medias[columnas_nan]

    # Crear carpetas de salida
    carpeta_archivo = os.path.join(
        carpeta_salida, nombre_base
    )

    carpeta_graficas = os.path.join(
        carpeta_archivo,
        f"graficas_{nombre_base}"
    )

    os.makedirs(
        carpeta_graficas,
        exist_ok=True
    )

    # --------------------------------------------------------
    # APLICAR FASTICA
    # --------------------------------------------------------

    X_limpio = X.copy()

    # Excluir canales constantes del cálculo de ICA
    indices_validos = np.where(
        np.std(X, axis=0) > 1e-12
    )[0]

    ica_aplicado = False

    try:

        if len(indices_validos) == 0:
            raise ValueError(
                "No hay canales con variación suficiente."
            )

        X_ica = X[:, indices_validos]

        n_componentes = min(
            5,
            X_ica.shape[1],
            X_ica.shape[0] - 1
        )

        if n_componentes < 1:
            raise ValueError(
                "No hay suficientes muestras para ICA."
            )

        ica = FastICA(
            n_components=n_componentes,
            random_state=42,
            max_iter=1000,
            tol=1e-4
        )

        with warnings.catch_warnings(record=True) as avisos:

            warnings.simplefilter("always")

            fuentes = ica.fit_transform(X_ica)

            for aviso in avisos:
                if issubclass(
                    aviso.category,
                    ConvergenceWarning
                ):
                    print(
                        "[WARNING] FastICA no convergió "
                        "completamente. Revisa los resultados."
                    )

        # Heurística experimental:
        # identificar el componente con mayor cuarto momento
        puntuaciones = np.mean(
            fuentes ** 4,
            axis=0
        )

        indice_artefacto = int(
            np.argmax(puntuaciones)
        )

        # Poner a cero el componente seleccionado
        fuentes_limpias = fuentes.copy()

        fuentes_limpias[:, indice_artefacto] = 0

        # Reconstruir los canales procesados
        reconstruccion = ica.inverse_transform(
            fuentes_limpias
        )

        X_limpio[:, indices_validos] = reconstruccion

        ica_aplicado = True

        print(
            "[INFO] FastICA aplicado correctamente."
        )

        print(
            "[INFO] Componente puesto a cero: "
            f"{indice_artefacto + 1}"
        )

    except Exception as error:

        print(
            f"[ERROR] No se pudo aplicar FastICA: {error}"
        )

        print(
            "[ERROR] No se generará un CSV limpio "
            "para este archivo."
        )

    # --------------------------------------------------------
    # EXPORTAR ÚNICAMENTE DATOS LIMPIOS
    # --------------------------------------------------------

    if ica_aplicado:

        datos_limpios = {
            "Muestra": np.arange(
                1, len(X_limpio) + 1
            )
        }

        for i, columna in enumerate(nombres_columnas):

            nombre_canal = (
                limpiar_nombre(columna)
                or f"Canal_{i + 1}"
            )

            # Solo se guarda la señal procesada
            datos_limpios[nombre_canal] = (
                X_limpio[:, i]
            )

        df_limpio = pd.DataFrame(
            datos_limpios
        )

        ruta_salida_csv = os.path.join(
            carpeta_archivo,
            f"Datos_Limpios_{nombre_base}.csv"
        )

        df_limpio.to_csv(
            ruta_salida_csv,
            index=False,
            encoding="utf-8-sig"
        )

        print(
            "[INFO] CSV limpio guardado: "
            f"{ruta_salida_csv}"
        )

    # --------------------------------------------------------
    # GENERAR GRÁFICAS GENERAL Y ZOOM
    # Las gráficas muestran las señales originales.
    # --------------------------------------------------------

    for i, columna in enumerate(nombres_columnas):

        nombre_canal = (
            limpiar_nombre(columna)
            or f"Canal_{i + 1}"
        )

        datos_serie = X_original[:, i]

        fin_zoom = min(
            300, len(datos_serie)
        )

        # Gráfica general
        plt.figure(figsize=(10, 4))

        plt.plot(
            datos_serie,
            color="blue"
        )

        plt.title(
            f"Señal EEG Original ({nombre_base}) - {columna}"
        )

        plt.xlabel("Muestras de tiempo (samples)")
        plt.ylabel("Amplitud / Potencia")

        plt.grid(True)
        plt.tight_layout()

        ruta_general = os.path.join(
            carpeta_graficas,
            f"canal_{i:02d}_{nombre_canal}_general.png"
        )

        plt.savefig(
            ruta_general,
            dpi=300
        )

        plt.close()

        # Gráfica zoom
        plt.figure(figsize=(10, 4))

        plt.plot(
            range(fin_zoom),
            datos_serie[:fin_zoom],
            color="red",
            marker="o",
            markersize=2
        )

        plt.title(
            f"Zoom EEG ({nombre_base}) - {columna}"
        )

        plt.xlabel("Muestras de tiempo (samples)")
        plt.ylabel("Amplitud / Potencia")

        plt.grid(True)
        plt.tight_layout()

        ruta_zoom = os.path.join(
            carpeta_graficas,
            f"canal_{i:02d}_{nombre_canal}_zoom.png"
        )

        plt.savefig(
            ruta_zoom,
            dpi=300
        )

        plt.close()

    print(
        f"[INFO] Gráficas guardadas en: {carpeta_graficas}"
    )

    return ica_aplicado


# ============================================================
# 4. SELECCIONAR ARCHIVOS CSV Y CARPETA DE SALIDA
# ============================================================

def seleccionar_documentos():

    ventana = tk.Tk()
    ventana.withdraw()

    try:

        ventana.attributes("-topmost", True)

        # Seleccionar uno o varios archivos
        archivos = filedialog.askopenfilenames(
            parent=ventana,
            title="Selecciona los archivos CSV de EEG",
            filetypes=[
                ("Archivos CSV", "*.csv"),
                ("Todos los archivos", "*.*")
            ]
        )

        if not archivos:

            messagebox.showinfo(
                "Selección cancelada",
                "No seleccionaste ningún archivo CSV.",
                parent=ventana
            )

            return

        # Seleccionar carpeta de destino
        carpeta_salida = filedialog.askdirectory(
            parent=ventana,
            title="Selecciona la carpeta de resultados"
        )

        if not carpeta_salida:

            messagebox.showinfo(
                "Selección cancelada",
                "No seleccionaste una carpeta de salida.",
                parent=ventana
            )

            return

        procesados = 0
        errores = []

        # Procesar archivos seleccionados
        for archivo in archivos:

            try:

                resultado = cargar_y_procesar_eeg(
                    archivo,
                    carpeta_salida
                )

                if resultado:
                    procesados += 1
                else:
                    errores.append(
                        f"{os.path.basename(archivo)}: "
                        "FastICA no se completó."
                    )

            except Exception as error:

                mensaje = (
                    f"{os.path.basename(archivo)}: {error}"
                )

                errores.append(mensaje)

                print(f"[ERROR] {mensaje}")

        # Resumen final
        resumen = (
            "Proceso terminado.\n\n"
            f"Archivos seleccionados: {len(archivos)}\n"
            f"Procesados con FastICA: {procesados}\n"
            f"Archivos sin CSV limpio: {len(errores)}\n\n"
            f"Carpeta de resultados:\n{carpeta_salida}"
        )

        if errores:

            resumen += (
                "\n\nDetalles:\n"
                + "\n".join(errores[:10])
            )

        print("\n[INFO] " + resumen)

        messagebox.showinfo(
            "Resultados del procesamiento",
            resumen,
            parent=ventana
        )

    finally:
        ventana.destroy()


# ============================================================
# 5. EJECUTAR EL EQUIPO SELECCIONADO
# ============================================================

def ejecutar_equipo(opcion):

    temas = {
        "1": ("Equipo 1", "Fast_ICA"),
        "2": ("Equipo 2", "Wavelet_ICA"),
        "3": ("Equipo 3", "Pasaband"),
        "4": ("Equipo 4", "InterferenciaModifier"),
        "5": ("Equipo 5", "ICA"),
        "6": ("Equipo 6", "ExtendInfomax_ICA"),
        "7": ("Equipo 7", "Artefacto_NoRaw")
    }

    if opcion not in temas:

        print(
            "[WARNING] Opción no válida. "
            "Selecciona un número del 0 al 7."
        )

        return

    equipo, tema = temas[opcion]

    print("\n" + "-" * 40)
    print(f"Seleccionaste: {equipo}")
    print(f"Tema: {tema}")
    print("-" * 40)

    if opcion == "1":

        seleccionar_documentos()

    else:

        print(
            f"[INFO] {tema} todavía no está implementado. "
            "Por ahora solo está disponible "
            "Equipo 1 - Fast_ICA."
        )


# ============================================================
# 6. PROGRAMA PRINCIPAL
# ============================================================

def main():

    while True:

        mostrar_menu()

        opcion = input(
            "Selecciona una opción (0-7): "
        ).strip()

        if opcion == "0":

            print(
                "Saliendo del programa. ¡Hasta pronto!"
            )

            break

        ejecutar_equipo(opcion)


if __name__ == "__main__":
    main()
