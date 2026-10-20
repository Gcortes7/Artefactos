# Copias para WEKA

Los mismos 40 archivos (20 originales + 20 limpios) convertidos a un formato
que el cargador CSV de WEKA acepta sin configurar nada.

**Estas copias no son el entregable.** Los archivos buenos están en
`../datos_originales/` y `../datos_limpios/`. Esta carpeta existe solo para
cargarlos en WEKA sin pelearse con el diálogo de opciones.

## Qué se cambió

| | |
|---|---|
| Separador | Punto y coma a coma en los 2 archivos de `2212605` |
| Nombres de atributo | Se quitó el espacio inicial (`" AF3/theta"` → `AF3/theta`) |
| Nombres de archivo | Normalizados a `<sujeto>_<tarea>.csv` en los originales |
| Filas incompletas | Eliminadas de los dos lados del par (21) |

Los valores numéricos no se tocaron. Los decimales ya venían con punto, así
que el cambio de separador no afecta ningún número.

## Por qué hubo que rellenar filas

**18 de los 20 archivos originales tienen una fila truncada, y siempre es la
última.** La grabación se cortó a media escritura. Los dos de `222260210` son
los únicos sin ella, que son justamente los duplicados.

En `2212605_PushPull.csv` la última fila está completamente vacía — mismo
fenómeno, otra forma.

WEKA aborta la carga cuando una fila trae menos campos que la cabecera
(*wrong number of values*).

Con autorización de la profesora, esas filas se eliminaron, y se eliminó
también la fila correspondiente del archivo limpio. Las 21 estaban al final de
su archivo, sin excepción.

El algoritmo descarta esos mismos registros antes de procesar, así que
`../datos_limpios/` también tiene 27,717. Los dos lados coinciden sin tener que
recortar nada aquí.

El motivo de descartarlos no es solo que WEKA no pueda cargarlos: una fila
truncada llega a la corrección con sus campos vacíos convertidos en `NaN`, y la
interpolación los rellena copiando la fila anterior. Una versión previa del
programa producía exactamente eso, con la última fila de `2212605_PushPull`
idéntica a la penúltima en las 70 variables.

## Verificación

Los 20 pares coinciden en filas y columnas: 27,717 registros de cada lado.

Revisión de los 40 archivos, con cero incidencias: sin BOM, sin atributos
repetidos ni con espacios, todas las filas del mismo ancho que su cabecera,
cero celdas vacías, cero valores `inf` o `nan`, cero texto en columnas
numéricas y cero filas repetidas consecutivas.

## Columnas eliminadas

`Timestamp`, `Hora` y `Fecha` ya no están en estas copias.

`Timestamp` identificaba la sesión de grabación: el clasificador podía acertar
reconociendo *cuándo* se grabó en vez de *qué* tarea era. Es fuga de
información. Con validación cruzada normal daba 88% de exactitud; con
leave-one-subject-out cayó a 51.5%, por debajo del 57.68% de ZeroR.

`Hora` y `Fecha` (solo en `2212605` y `222260210`) entraban como nominales con
cientos de valores distintos y no aportaban nada.

Al quitarlas, **los 40 archivos quedan con las mismas 70 columnas** —14 canales
× 5 bandas, en el mismo orden— y desaparece la diferencia 71/73 del origen.

## No hay atributo de clase

Estos archivos son 70 atributos numéricos y nada que predecir, así que sirven
para *Preprocess* y *Visualize* pero no para *Classify* tal cual. Para
clasificar hay que añadir la columna de tarea (LeftRight / PushPull) y unir los
archivos.

## Comprobación

Los 40 archivos pasan sin incidencias: sin BOM, sin saltos CRLF, sin atributos
repetidos ni con espacios, todas las filas del mismo ancho que su cabecera,
cero celdas vacías, cero valores `inf` o `nan`, cero texto en columnas
numéricas y **cero filas idénticas a la anterior**.

Esta última comprobación importa: una fila que repite exactamente a la previa
en las 70 variables es la firma de un registro copiado en vez de medido. En los
20 archivos originales no hay ninguna, y tras descartar los registros
incompletos tampoco queda ninguna en los limpios.
