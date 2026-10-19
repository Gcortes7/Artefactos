# Equipo 7 — Artefacto_NoRaw

**Integrantes:** Nestor Gael Lara Juárez · Diego Guzmán Rodríguez
**Materia:** Inteligencia Artificial — 5801 IA 2026-2
**Profesora:** Griselda Cortes Barrera

---

## Qué resuelve este algoritmo

ICA, FastICA, WaveletICA y el filtrado pasa-banda operan sobre la **señal cruda
multicanal en el dominio del tiempo**: descomponen la mezcla de fuentes y
reconstruyen la señal sin las componentes contaminadas. Los cuatro necesitan la
forma de onda.

Los archivos de esta práctica no la contienen. Traen **potencia de banda ya
calculada** por la diadema Emotiv: 14 canales × 5 bandas. El cálculo aplica FFT,
promedia la energía dentro de cada banda y descarta la fase, de modo que es
**irreversible**. Sin forma de onda no hay mezcla lineal de fuentes que ICA pueda
separar, y un filtro pasa-banda no tiene nada que filtrar porque el eje de
frecuencia ya se colapsó en cinco números por canal.

El tema asignado al equipo 7 es exactamente ese: qué hacer cuando los datos ya no
son crudos. La respuesta es trabajar en el **espacio de características**, con
detectores específicos de la firma espectral de cada artefacto.

---

## Los tres artefactos

| Artefacto | Dónde se manifiesta | Cómo se detecta |
|---|---|---|
| **Ocular** | Frontales (AF3, AF4, F7, F8), bandas theta y alpha | Gradiente frontal-posterior: el dipolo córneo-retiniano decae con la distancia al ojo |
| **Muscular** | Periféricos (T7, T8, F7, F8), bandas betaH y gamma | Proporción de potencia alta: el EMG se concentra por encima de 20 Hz |
| **Interferencia** | Cualquier canal, de forma puntual | Ceros exactos y saltos locales extremos, celda por celda |

Sobre la interferencia: **no es ruido de línea**. La diadema aplica un notch dual
de 50/60 Hz por hardware antes de transmitir, y gamma llega solo hasta ~45 Hz. Lo
que sí aparece es pérdida de contacto del electrodo.

---

## Tres decisiones de diseño

**1. Los índices son diferencias de logaritmos, no cocientes.**
Un cociente se dispara cuando el denominador se acerca a cero. En las pruebas
llegó a reportar que el artefacto había aumentado un 700% cuando en realidad las
dos potencias habían bajado. La escala logarítmica es además la natural para
potencia de banda, que sigue una distribución log-normal.

**2. Se exigen dos condiciones para marcar una muestra.**
Que la razón esté elevada **y** que la potencia que ese artefacto aporta también
lo esté. Una razón puede subir porque el numerador creció o porque el denominador
colapsó, y solo lo primero es un artefacto fisiológico: el parpadeo y el músculo
*añaden* potencia, nunca la retiran.

**3. Cada detección se expande ±8 muestras.**
Emotiv calcula la potencia con una ventana de 256 muestras desplazada 16 a la vez,
lo que deja un **94% de solapamiento** entre registros consecutivos. Una ráfaga
instantánea de artefacto contamina del orden de 16 registros.

---

## Qué conserva el archivo procesado

- El **mismo número de registros** que el original
- Las mismas columnas, en el mismo orden
- El separador original (coma o punto y coma)
- Las marcas de tiempo y los metadatos

En las muestras marcadas, y solo en las variables que ese artefacto afecta, el
valor se sustituye por **interpolación lineal entre las muestras limpias vecinas**.
Se prefiere a imputar la media porque la potencia de banda varía lentamente dentro
de una sesión: los vecinos inmediatos son mejor estimador que cualquier
estadístico global.

---

## Uso

### Desde el Menu.ipynb

Ejecuta la celda que carga el módulo, corre `main()` y elige la opción **7**.
El menú del equipo permite ver la información del algoritmo y **limpiar
cualquier archivo que subas**, con los mismos parámetros que usamos nosotros.

### Como módulo

```python
import artefacto_noraw as e7

e7.info()                          # información del equipo y del algoritmo
e7.limpiar_archivo("datos.csv")    # limpia un archivo
e7.limpiar_carpeta("./mis_datos")  # limpia todos los CSV de una carpeta
e7.menu()                          # menú interactivo (funciona en Colab)
```

### Desde la terminal

```bash
python3 artefacto_noraw.py archivo.csv
python3 artefacto_noraw.py --carpeta ./datos --salida ./limpios
python3 artefacto_noraw.py --info
```

El archivo de salida sigue la nomenclatura `<sujeto>_<tarea>_NoRaw_Clean.csv`.

---

## Parámetros

| Parámetro | Valor | Por qué |
|---|---|---|
| Umbral z robusto | 3.5 | Convencional con MAD; equivale a ~3.5 desviaciones |
| Condición de potencia | 2.0 | Más bajo porque es apoyo, no el criterio principal |
| Umbral de interferencia | 6.0 | Más alto: la detección por celda hace muchas más pruebas |
| Expansión de máscara | ±8 muestras | Cubre el solapamiento del 94% |
| Expansión de interferencia | ±2 muestras | Los fallos de electrodo son puntuales |

El z-score se calcula con **mediana y MAD** y no con media y desviación estándar,
porque los propios artefactos inflan esas dos estadísticas y elevarían el umbral
justo donde el detector debería ser más sensible.

---

## Resultados sobre los 20 datasets

| Indicador | Resultado |
|---|---|
| Registros | 27,738 — conservados al 100% |
| Celdas corregidas | 5.3% en promedio |
| Celdas sin señal (cero exacto) | 7,264 → 0 |
| Reducción del artefacto ocular | 88.8% (mediana, donde se detectó) |
| Correlación con el original | 0.956 (media, escala logarítmica) |
| Validación | 19 de 20 aprobados |

El archivo restante, `222260245_PullPush`, supera cuatro de los cinco criterios:
la corrección funcionó, pero es el más contaminado del lote y el volumen de
intervención baja la correlación a 0.884. Se deja marcado para revisión en lugar
de ajustar el umbral hasta que pase.

### Limitación conocida

El método detecta artefactos **episódicos**, no contaminación sostenida. El umbral
se calcula con la mediana del propio archivo, así que si un sujeto mantuvo la
mandíbula apretada durante toda la grabación, esa contaminación ya forma parte de
la mediana y no hay pico que detectar.

---

## Contenido de esta carpeta

```
Equipo7_NoRaw/
├── artefacto_noraw.py          El algoritmo, ejecutable e importable
├── README.md                   Este archivo
├── datos_limpios/              Los 20 datasets procesados
│   ├── *_NoRaw_Clean.csv
│   └── reporte_limpieza.csv
└── reportes/                   Excel con tablas y gráficas
```
