# Revisión del modelo

Se revisaron los dos scripts. Las modificaciones están en `football_poisson.py`, el modelo con estadísticas. El script sin estadísticas conserva su comportamiento.

## Cambios

- Probabilidades 1X2 calculadas con Skellam y corrección Dixon–Coles analítica. Se elimina el truncamiento en diez goles; las distribuciones marginales y la suma de probabilidades se conservan. El argumento `max_goals` se mantiene por compatibilidad.
- `--auto-tune` permite seleccionar regularización, recencia y corrección de empates en dos ventanas cronológicas dentro del entrenamiento (60–80 % y 80–100 %). Se comparan los mismos partidos y se mantiene la configuración original cuando la ventaja en log loss es menor que 0,002. El holdout externo nunca participa en esta selección.
- El modelo guardado registra los parámetros, fechas, tamaños de validación y resultados de candidatos. El ajuste automático queda desactivado por defecto: todavía no acredita una mejora externa.

## Prueba externa

Entrenamiento hasta 2024 y temporada de prueba 2025, usando bases locales. Se evalúan todos los partidos con predicción, incluyendo los de baja confianza. Menor log loss y Brier es mejor.

| Liga | Variante | Partidos | Acierto | Log loss | Brier |
|---|---|---:|---:|---:|---:|
| Colombia | Original | 409 de 450 | 48,90 % | 1,02098 | 0,61307 |
| Colombia | Ajuste automático | 409 de 450 | 47,92 % | 1,02113 | 0,61324 |
| Inglaterra | Original | 380 de 380 | 49,47 % | 1,03873 | 0,62356 |
| Inglaterra | Ajuste automático | 380 de 380 | 49,47 % | 1,03873 | 0,62356 |

Colombia seleccionó alpha=0,1, decay=0,995 y Dixon–Coles activo. Su validación interna pasó de 1,0304 a 1,0233, pero esa ventaja no se mantuvo en 2025. Inglaterra conservó los parámetros originales. No se afirma una mejora de precisión: hace falta evaluar más temporadas sin ajustar decisiones a sus resultados. Los modelos joblib existentes no se sobrescribieron.

El 60 % usado para filtrar partidos es un umbral de confianza del modelo, no un porcentaje de aciertos garantizado. En Colombia faltan predicciones para 41 partidos; la evaluación no representa esos casos. Los resultados de 2025 pueden abarcar encuentros de 2026 cuando la liga usa temporadas entre dos años.

## Uso

```powershell
python football_poisson.py backtest --country Colombia --train-through 2024 --test-season 2025 --auto-tune --output model_review/colombia_tuned.csv
python football_poisson.py train --country Colombia --auto-tune --model model_review/colombia_experimental.joblib
python -m unittest test_football_model.py
```

Cuatro pruebas verifican equivalencia con matrices de marcadores amplias, simetría, rechazo de tasas inválidas y separación cronológica de las ventanas. CSV, logs y métricas JSON están en `model_review/`.
