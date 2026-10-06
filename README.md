# scintia-s4 — Pronóstico de centelleo ionosférico S4 (Tesis MCC UNI–IGP)

Código ordenado del sistema de pronóstico. **La lógica vive en `src/s4/`; los notebooks solo la llaman.**
Cada número de la tesis debe salir de una función de este paquete y quedar en `experiments/registro_*.csv`.

## Instalación (RTX 4080, Linux)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q tests          # 9 pruebas: fuga, ventanas, causalidad, métricas, LSTM y PatchTST
```

Editar `rutas.base` en `config/ablacion_2025_2026.yaml` y copiar los datos a `data/`:
- `data/processed/df_max_s4_reconstruido.csv` (salida de la reconstrucción)
- `data/raw/DF_FINAL_JICAMARCA_31072026.csv` (precursores)

## Orden de ejecución

| # | Notebook | Responde a | ¿Entrena? |
|---|---|---|---|
| 01 | `01_reconstruccion` | Yuri: sustento de la interpolación | No |
| 02 | `02_split_y_eda` | Alania 5.1: diagrama Out-of-Time. **Copiar las fechas de corte al YAML** | No |
| 04 | `04_reevaluacion_modelo_produccion` | Alania: "¿15 % de precisión?", persistencia con significancia | No |
| 03 | `03_acf_y_lookback` | Yuri: "sustentar que las pruebas son las mejores" | Sí |
| 05 | `05_ablacion` | Alania 5.4 + pregunta sobre ROTI | Sí |
| 06 | `06_sensibilidad_alpha_beta` | Alania: "¿por qué α y β?" | Sí |
| 07 | `07_patchtst_vs_lstm` | Alania 5.2: Transformer, Recall y latencia | Sí |
| 08 | `08_explicabilidad` | Alania: feature importance / saliencia | Sí (1 modelo) |
| 09 | `09_resultados_finales` | Única evaluación en Test + exportación a la app | Sí |

Regla: **todas las decisiones se toman en Validación**. Test solo se toca en 04 (modelo ya entrenado) y en 09.

## Módulos

| Módulo | Contenido |
|---|---|
| `config.py` | YAML único, variantes para grids, huella reproducible |
| `io.py` | Lectura del S4 reconstruido (con bandera `reconstruido`) y de precursores |
| `reconstruction.py` | Fix0–Fix4 y auditor de episodios (migrado sin cambios) |
| `features.py` | Codificación cíclica, **precursores causales**, conjuntos de la ablación |
| `splits.py` | Eventos, búsqueda del corte, partición por día ionosférico, auditoría de fuga |
| `windows.py` | Escalado fit-en-Train, ventanas sin OOM, objetivos comunes entre lookbacks |
| `losses.py` | Weighted Focal MSE (fórmula exacta en el docstring) |
| `models/` | Persistencia, climatología, LSTM simple/apilado/bidireccional, PatchTST |
| `train.py` | Entrenamiento, predicción en escala física, latencia CPU/GPU |
| `evaluate.py` | Regresión + alerta (POD, Precisión, FAR, CSI, HSS) + fases (inicio/sostenido) |
| `stats.py` | Bootstrap por días y Diebold–Mariano |
| `explain.py` | Importancia por permutación y gradientes integrados |
| `registry.py` | Registro de experimentos con huella de configuración |
| `pipeline.py` | Orquestación: `preparar_datos`, `correr_variante`, `calibrar_umbral` |

## Cambios metodológicos respecto a los notebooks anteriores

1. **Índices geomagnéticos causales.** En el CSV multivariable, Dst y AE están interpolados linealmente
   entre horas: el minuto h:35 contiene información de la hora h+1 (fuga de hasta ~60 min).
   `features.causalizar_indice` usa en cada minuto solo el último valor ya disponible.
2. **Corte a las 12 UT (07 LT).** Con corte a medianoche UT (19 LT) una noche de centelleo podía quedar
   partida entre Train y Validación. La auditoría ahora detecta eventos que cruzan un corte.
3. **Bootstrap por días** en lugar de por ventanas (las ventanas consecutivas no son independientes) y
   prueba de **Diebold–Mariano**.
4. **Métricas de alerta y por fase.** El RMSE en eventos no se usa solo para seleccionar: premia sobreestimar.
5. **Umbral de alerta calibrado en Validación** (máximo HSS), aplicado sin cambios en Test.
6. **Minutos reconstruidos excluidos** de todas las métricas.
7. **Sin `abs()` en la evaluación**: un S4 negativo predicho es un error del modelo y debe contarse.
8. **Baseline adicional**: climatología diaria (media de S4 por minuto del día, solo con Train).

## Correspondencia con el código anterior

| Antes (notebook, sección) | Ahora |
|---|---|
| TESIS_MCC_01102026 §4 `identify_events` | `splits.identificar_eventos` |
| §6–8 `find_temporal_split`, `create_temporal_split`, `validate_temporal_split` | `splits.buscar_corte_temporal`, `particionar`, `auditar_fuga` |
| §10–12 cíclicas y escalado | `features.agregar_ciclicas`, `windows.ajustar_escalador` |
| §14–15 `compute_valid_starts`, `build_windowed_dataset` | `windows.inicios_validos`, `Particion.dataset` |
| §16 `baseline_persistencia` | `models.persistencia` |
| §17 `evaluar_predicciones` | `evaluate.evaluar` |
| §20 `weighted_focal_mse_multistep` | `losses.weighted_focal_mse` |
| §25 `construir_lstm_arquitectura` | `models.construir_lstm` |
| §24 `bootstrap_rmse_evento` | `stats.bootstrap_bloques` |
| PRONOSTICO_S4_documentado (todo) | `reconstruction.py` |
