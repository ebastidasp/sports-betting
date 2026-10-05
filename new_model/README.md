# Modelo nuevo entrenado desde cero

`football_model_v2.py` es independiente de los programas anteriores. No carga sus modelos, no importa sus fórmulas y no descarga información: lee exclusivamente las tablas `fixtures`, `team_stats` y `app_config` de los archivos SQLite existentes, en modo de solo lectura.

Se entrenó con **50.785 partidos finalizados de 11 bases**, con resultados disponibles antes del 30 de septiembre de 2026 a las 00:00 de Colombia. Las bases vacías se omiten. Los encuentros con prórroga o penaltis se excluyen para que el objetivo corresponda al resultado en tiempo reglamentario. La ausencia de estadísticas de tiros no elimina un partido.

## Qué aprende

Aprende probabilidades de victoria local, empate y victoria visitante a partir de 167 variables previas al partido: goles anotados y recibidos, puntos recientes, rendimientos locales y visitantes, fuerza Elo, descanso, tiros, posesión, córners y xG cuando están disponibles. Las estadísticas se suavizan con horizontes de cinco y veinte partidos y una pequeña regularización para equipos con poco historial. La disponibilidad de estadísticas también se representa en las variables.

Se compararon regresión logística, árboles potenciados por gradiente, una referencia Poisson entrenada desde cero y combinaciones de sus probabilidades. La selección por acierto en 2024 eligió **HistGradientBoostingClassifier: 120 iteraciones, siete hojas por árbol, tasa de aprendizaje 0,05**. No se eligió el modelo usando los resultados de 2025. No genera un marcador exacto.

## Evaluación independiente

Los candidatos se entrenaron con resultados disponibles antes del 1 de enero de 2024; se seleccionaron con partidos del año calendario 2024. Después se reajustó la configuración elegida con los resultados disponibles antes del 1 de enero de 2025 y se evaluó en el año calendario 2025. Las temporadas que cruzan dos años se separan por fecha real del partido.

Durante esa evaluación, cada predicción usa el historial disponible en su fecha. El resultado de un partido se considera disponible tres horas después de su inicio, una aproximación conservadora porque el caché no guarda la hora de finalización. Los partidos simultáneos no conocen los resultados de los demás. Imputadores y codificaciones se ajustan únicamente con el conjunto de entrenamiento.

En los **3.249 partidos comunes** con el modelo anterior, entrenado nuevamente con el mismo corte temporal:

| Medida | Modelo anterior | Modelo nuevo |
|---|---:|---:|
| Acierto 1X2 | 48,78 % | **50,20 %** |
| Log loss (menor es mejor) | 1,02139 | **1,01641** |
| Brier (menor es mejor) | 0,61155 | **0,60829** |

La diferencia de acierto es **+1,42 puntos porcentuales**, equivalente a 46 aciertos adicionales. Un bootstrap por semanas proporciona un intervalo exploratorio del 95 % de +0,23 a +2,65 puntos. Este resultado corresponde a la muestra evaluada y no garantiza el rendimiento en otras temporadas.

| País | Partidos comunes | Anterior | Nuevo |
|---|---:|---:|---:|
| Argentina | 471 | 41,19 % | 44,16 % |
| Chile | 210 | 48,57 % | 50,00 % |
| Colombia | 409 | 48,90 % | 49,88 % |
| Inglaterra | 378 | 53,97 % | 54,23 % |
| Alemania | 308 | 50,65 % | 49,68 % |
| Irlanda | 152 | 42,76 % | 46,71 % |
| Italia | 367 | 49,86 % | 52,86 % |
| Corea del Sur | 89 | 43,82 % | 46,07 % |
| España | 370 | 54,32 % | 55,95 % |
| Estados Unidos | 495 | 48,69 % | 49,09 % |

La mejora no es uniforme: Alemania baja en acierto y algunas ligas empeoran en log loss. Los porcentajes de Inglaterra corresponden al año calendario 2025 y no son directamente comparables con el backtest anterior de la temporada 2025.

El nuevo modelo también predice los **3.725 partidos de primera división** disponibles en 2025, incluyendo Islandia y encuentros que el anterior omitía. En esa población completa obtiene 49,72 % de acierto; la referencia Poisson independiente obtiene 49,02 %. No se restringen las métricas a partidos de alta confianza.

## Usar el modelo ya entrenado

Desde la carpeta del proyecto:

```powershell
python football_model_v2.py predict --country colombia --home Millonarios --away "Santa Fe"
python football_model_v2.py predict --country england --home Arsenal --away Chelsea
```

Se puede usar el nombre o el ID del equipo. `--league-id` permite seleccionar otra división configurada. Para comprobar una fecha concreta, `--as-of` debe incluir su zona horaria. Se rechazan fechas anteriores al corte de entrenamiento del modelo utilizado.

## Entrenar nuevamente

```powershell
# Todas las bases SQLite de la carpeta
python football_model_v2.py train

# Modelo independiente para un país
python football_model_v2.py train --db colombia.sqlite3 --output new_model_colombia

# Reproducir el entrenamiento entregado
python football_model_v2.py train --as-of 2026-09-30T05:00:00+00:00

# Dar prioridad a la calidad de probabilidades al seleccionar
python football_model_v2.py train --selection-metric log_loss --output new_model_logloss
```

`--validation-year` y `--test-year` permiten cambiar los cortes; el año de validación debe ser anterior al de prueba. Por defecto el entrenamiento final utiliza todos los resultados disponibles, incluido 2025, **después** de evaluar la configuración congelada. Por eso el archivo final sirve para predicciones futuras; `evaluation_model.joblib` conserva el ajuste anterior a 2025 para reproducir la evaluación histórica.

## Archivos

- `football_model.joblib`: nuevo modelo final entrenado.
- `evaluation_model.joblib`: modelo entrenado antes del año de prueba.
- `evaluation.json`: candidatos, cortes temporales, métricas y fuentes.
- `test_predictions.csv`: predicciones de los 3.725 partidos de prueba.
- `comparison_previous.json` y `comparison_summary.json`: comparación por país y agregada con el programa anterior.
- `comparison_predictions.csv`: probabilidades de ambos modelos para los mismos partidos.

`compare_football_models.py` es un programa de auditoría separado que importa el modelo anterior para compararlo; el entrenamiento y las predicciones del nuevo modelo no dependen de él.

## Verificación

```powershell
python -m unittest test_football_model.py test_football_model_v2.py
```

Pasaron 14 pruebas, diez específicas del modelo nuevo. Comprueban separación entre predicción y resultado, partidos simultáneos, demora de disponibilidad, corte histórico, datos faltantes, lectura sin modificar SQLite, suavizado incremental, probabilidades y guardado/carga. También se comprobó una predicción real desde el modelo final guardado.
