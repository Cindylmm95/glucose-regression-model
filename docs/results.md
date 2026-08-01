# Resultados técnicos

## Dataset CGM

- 64 participantes.
- 864 lecturas originales por participante.
- 53,760 registros model-ready.
- 24 lecturas históricas por observación.
- 51 características.
- Horizonte aproximado de 5 minutos.

## División V2

| Conjunto | Participantes | Registros |
|---|---:|---:|
| Entrenamiento | 44 | 36,960 |
| Validación externa | 10 | 8,400 |
| Prueba reservada | 10 | 8,400 |

La división se realizó por participante para evitar fuga de información.

## IBM AutoAI

Modelo seleccionado:

`Glucose_Profile_V2_Ridge_Model`

| Métrica | Holdout | Cross-validation |
|---|---:|---:|
| RMSE | 0.785 | 0.814 |
| MAE | 0.617 | 0.627 |
| R² | 0.998 | 0.998 |

## Validación externa

| Métrica | Resultado |
|---|---:|
| Dentro de 2.5 mg/dL | 99.39% |
| Dentro de 5 mg/dL | 99.92% |
| MAE | 0.675 mg/dL |
| Error absoluto P95 | 1.559 mg/dL |

La prueba online con Python envió tres registros al endpoint y recibió una respuesta HTTP 200. Los tres resultados quedaron dentro de 2.5 mg/dL.

## Consideraciones

La alta precisión se relaciona con el horizonte corto y la autocorrelación de las series CGM. El modelo no representa una herramienta clínica.
