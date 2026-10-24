# Datos originales (antes del tratamiento)

Los 20 datasets EEG tal como se recibieron, **sin ninguna modificación**.
Esta carpeta existe para que cualquiera pueda reproducir la limpieza desde
el punto de partida y comparar el antes contra el después.

## Qué hay aquí

| | |
|---|---|
| Archivos | 20 CSV (10 sujetos × 2 tareas) |
| Registros | 27,738 en total, de los cuales 21 están incompletos |
| Variables de señal | 14 canales × 5 bandas = 70 columnas |
| Contenido | Potencia de banda calculada por la diadema Emotiv, **no señal cruda** |

`../INDICE_ORIGINALES.csv` lista, por archivo: sujeto, tarea, registros,
columnas, separador, hash MD5 y con qué archivo limpio corresponde. Está un
nivel arriba a propósito: esta carpeta contiene únicamente los 20 archivos
de origen, para que procesarla completa no recoja nada que no sea señal.

## Los nombres están tal cual

No se renombró nada. Por eso hay inconsistencias en el origen:

- `212260569PushPull.csv` sin guion bajo
- `212260666_leftRight.csv` en minúscula
- `212260672_LeftRigth.csv` y `222260671_LeftRigth.csv` con *Rigth* en vez de *Right*
- La tarea aparece como `PushPull` en unos archivos y `PullPush` en otros

Se conservaron a propósito. Renombrarlos aquí ocultaría una característica
real del lote de datos, y el código ya normaliza los nombres a la salida.

## El separador no es el mismo en todos

18 archivos usan coma. Los dos de `2212605` usan punto y coma y además
traen 73 columnas en vez de 71. El código detecta el separador por archivo
y lo conserva en la salida.

## Un archivo está duplicado

`222260210_PullPush.csv` es una **copia byte por byte** de
`222260210_LeftRight.csv` — mismo MD5 (`489de797535a5684bfb230b85860ac5c`).
Ese sujeto no tiene grabación real de Push-Pull.

Es un problema de captura, no de limpieza: ningún algoritmo lo resuelve.
Se documenta aquí porque afecta cualquier análisis que compare las dos
tareas, y porque es la razón por la que la validación reporta 19 de 20
archivos y no 20 de 20.

## Reproducir la limpieza

Desde la raíz del repositorio:

```
python3 Equipo7_NoRaw/artefacto_noraw.py --carpeta Equipo7_NoRaw/datos_originales --salida salida_propia
```

El resultado debe coincidir con `../datos_limpios/`.
