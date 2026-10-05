# Modelo separado de córners

El programa es **`football_corners.py`**, un archivo independiente de los modelos de resultados. No los importa, no carga sus modelos y no realiza solicitudes de red. Lee las bases SQLite locales en modo de solo lectura y guarda sus propios modelos en `corners_model/`.

## Predicciones

- Córners esperados del local, del visitante y del partido completo.
- Probabilidad de cada total de 0 a 9 córners y de más de 9 (10 o más). `predict` muestra esta tabla automáticamente.
- Intervalos aproximados de predicción del 90 % para cada cantidad.
- Probabilidades de superar o quedar por debajo de un total configurable, por defecto 9,5.

La media puede ser decimal: por ejemplo, 9,23 córners esperados no significa que el resultado exacto será nueve. El intervalo describe la variabilidad de un partido, no la incertidumbre sobre la media.

## Estadísticas utilizadas

Se utilizan córners a favor y en contra, tiros totales, a puerta, fuera, bloqueados, dentro y fuera del área, posesión, pases, precisión de pase y xG cuando existen. También se incorporan goles históricos, Elo y días de descanso.

Estas estadísticas se calculan exclusivamente a partir de partidos anteriores, en ventanas exponenciales con semividas de cinco y veinte partidos. Se distinguen el rendimiento general y el correspondiente a local/visitante, y se combinan los córners producidos por un equipo con los concedidos por su rival. Se usan 255 variables, incluidas las de cobertura de estadísticas.

Se aceptan tanto los nombres de `STAT_ALIASES` como las claves normalizadas guardadas en los SQLite. Un córner faltante se conserva como dato desconocido; nunca se convierte en cero. Los equipos con poco historial reciben estimaciones regularizadas hacia el promedio de su liga.

## Entrenamiento y evaluación

El entrenamiento final empleó **71.322 conteos observados de córners por equipo**. Los historiales se reconstruyeron a partir de 50.785 partidos FT de 11 bases; una misma fecha aporta hasta dos conteos observados, uno por equipo. Islandia no tiene conteos de córners en el caché, por lo que no aporta objetivos supervisados.

Se entrenaron candidatos con resultados anteriores a 2024, se seleccionaron con partidos del año calendario 2024 y se evaluaron en 2025 sin usar ese año para seleccionar parámetros. Se compararon promedios históricos, regresión Poisson regularizada, árboles potenciados con pérdida Poisson y mezclas de sus predicciones.

La configuración elegida fue **75 % regresión Poisson (alpha=0,1) y 25 % promedio suavizado del enfrentamiento**. La selección minimiza el error absoluto medio del total de córners. Se recalculó el ajuste antes de 2025 para la prueba y después se entrenó el modelo final con todos los conteos disponibles.

Un resultado se considera disponible tres horas después del inicio del partido: el caché no registra la hora exacta de finalización. Las estadísticas del partido que se predice no entran en sus variables. Se excluyen encuentros con prórroga/penaltis para mantener el objetivo en tiempo reglamentario.

De los 3.725 partidos de primera división disponibles en el año calendario 2025, **3.425** tienen ambos conteos de córners y permiten evaluar el total. Las métricas restantes se calculan solo sobre valores observados.

| Medida del total | Promedio histórico de liga | Modelo |
|---|---:|---:|
| Error absoluto medio (MAE) | 2,695 | **2,657** |
| Raíz del error cuadrático medio (RMSE) | 3,355 | **3,310** |
| Desviación Poisson | 1,210 | **1,178** |
| Sesgo medio (predicción − realidad) | +0,206 | **+0,021** |

El error absoluto medio mejora aproximadamente un **1,39 %** frente a ese promedio simple. La mejora es pequeña; esto no equivale a acertar el número exacto de córners en cada partido.

La dispersión se estima con los errores de validación de 2024 mediante una distribución binomial negativa, o Poisson si no hay sobredispersión. En 2025, el intervalo nominal del 90 % para el total cubrió el **92,32 %** de los conteos observados. Son intervalos aproximados, y las probabilidades por línea no se han evaluado como una estrategia de apuestas.

## Uso

Desde la carpeta del proyecto:

```powershell
python football_corners.py predict --country colombia --home Millonarios --away "Santa Fe"
python football_corners.py predict --country england --home Arsenal --away Chelsea --line 10.5
```

La línea puede ser entera o fraccionaria. Para una línea entera se muestra también la probabilidad de igualdad. Se pueden identificar equipos por nombre o ID y elegir otra división con `--league-id`.

```powershell
# Entrenar con todos los SQLite de la carpeta
python football_corners.py train

# Entrenar solo con Colombia, guardando otro modelo
python football_corners.py train --db colombia.sqlite3 --output corners_colombia
python football_corners.py predict --model corners_colombia/corners_model.joblib --country colombia --home Millonarios --away "Santa Fe"

# Cambiar los años de validación y prueba
python football_corners.py train --validation-year 2023 --test-year 2024 --output corners_test_2024
```

`--as-of` acepta una fecha ISO con zona horaria. En entrenamiento limita los resultados disponibles; en predicción debe ser igual o posterior al corte del modelo. El corte exacto del modelo entregado está en `evaluation.json`.

## Archivos y comprobaciones

- `corners_model.joblib`: modelo final entrenado.
- `evaluation_model.joblib`: ajuste anterior a 2025 para reproducir predicciones históricas.
- `evaluation.json`: parámetros, variables, fuentes y métricas, también por país y por equipo local/visitante.
- `test_predictions.csv`: conteos reales, esperados y probabilidades del conjunto de prueba. Los objetivos faltantes quedan vacíos.

Pasaron las diez pruebas de `test_football_corners.py`: comprueban que las estadísticas actuales no alteran la predicción del propio partido, separación temporal, partidos simultáneos, conservación de ceros reales, datos faltantes, alias, distribuciones de conteos, guardado/carga y lectura sin modificar los datos SQLite. También se comprobó una predicción desde el modelo final guardado.

```powershell
python -m unittest test_football_corners.py
```

## Active version restored

The default model again uses total passes and pass accuracy, including opponent histories. The saved model, pre-2025 evaluation fit and test predictions were restored from `corners_model_with_passes/`. The version without passes is preserved in `corners_model_without_passes/`. The comparative analysis files are historical comparisons; `evaluation.json` describes the active model. All 11 current tests pass. The command also prints probabilities for totals 0 through 9 and more than 9.
