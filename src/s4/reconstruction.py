from __future__ import annotations
"""
Reconstrucción de la serie de S4 (Fix0 -> Fix1 -> filtro -> Fix3 -> Fix4).

Código migrado SIN cambios de lógica desde
PRONOSTICO_S4_documentado_05102026_1212.ipynb (celdas 3, 9, 11, 14, 16, 18, 20).
La fundamentación teórica (AR bidireccional + bootstrap de residuos) está en
las celdas markdown 7.x de ese notebook y debe citarse en el Cap. III.

Nota metodológica: la reconstrucción usa vecinos a ambos lados del hueco
(es bidireccional). Por eso, en evaluación, los minutos reconstruidos se
EXCLUYEN de las métricas de Test (ver s4.evaluate, parámetro `mask_valid`).
"""
# ======================================================================
# Celda original 3
# ======================================================================
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg

# Muestras esperadas en un día completo: 24 h * 60 min = 1440 puntos.
# Es la referencia de "100 % de cobertura" en todo el pipeline.
PUNTOS_DIA = 1440
S4_MAX_FISICO = 1.5


pd.options.mode.chained_assignment = None  # evita SettingWithCopyWarning por los .copy() intencionales

class CargadorDatosS4:
    '''Carga y limpieza inicial del dataset crudo de S4 (migrado sin cambios).'''

    def __init__(self, ruta: str, columnas_a_eliminar: Optional[list[str]] = None,
                 montar_drive: bool = True):
        self.ruta = ruta
        self.columnas_a_eliminar = columnas_a_eliminar or ["Datetime", "ID_Satelite", "Azimuth", "Elevacion"]
        self.montar_drive = montar_drive

    def _montar_drive(self) -> None:
        try:
            from google.colab import drive  # type: ignore
            drive.mount("/gdrive")
        except ImportError:
            warnings.warn("No se pudo importar google.colab.drive; se asume entorno no-Colab.")

    def cargar(self) -> pd.DataFrame:
        if self.montar_drive:
            self._montar_drive()
        df = pd.read_csv(self.ruta)
        df["datetime"] = pd.to_datetime(df["Datetime"])
        df = df.set_index("datetime").sort_index()
        columnas_presentes = [c for c in self.columnas_a_eliminar if c in df.columns]
        return df.drop(columns=columnas_presentes)
# ======================================================================
# Celda original 9
# ======================================================================
def cobertura_diaria(df: pd.DataFrame, puntos_dia: int = PUNTOS_DIA,
                     columna: str = "S4") -> pd.DataFrame:
    '''
    Porcentaje de minutos con S4 válido por día calendario.
    Cuenta valores no-NaN de `columna`, de modo que funciona tanto con
    df_max_s4_all_*.csv (solo minutos con dato) como con la versión en
    malla df_max_s4_all_grid_*.csv (minutos sin dato = NaN).
    '''
    df_ordenado = df.sort_index()
    validos = pd.to_numeric(df_ordenado[columna], errors="coerce").notna()
    resumen = validos.groupby(df_ordenado.index.normalize()).sum().rename("puntos").to_frame()
    resumen["puntos"] = resumen["puntos"].astype(int)
    resumen = resumen[resumen["puntos"] > 0]
    resumen["puntos_faltantes"] = puntos_dia - resumen["puntos"]
    resumen["porcentaje"] = (resumen["puntos"] / puntos_dia * 100).clip(upper=100)
    resumen["completo_100"] = resumen["puntos"] == puntos_dia
    resumen["anio"] = resumen.index.year
    return resumen[["anio", "puntos", "puntos_faltantes", "porcentaje", "completo_100"]]


def resumen_dias_sobre_umbral(resumen: pd.DataFrame, umbral: float = 90) -> dict:
    '''
    Cuenta cuántos días superan un umbral de cobertura (ej. >90 %) y devuelve
    un pequeño resumen imprimible.
    '''
    dias_sobre = (resumen["porcentaje"] > umbral).sum()
    total_dias = len(resumen)
    porcentaje_sobre = dias_sobre / total_dias * 100 if total_dias else 0.0
    return {
        "dias_sobre_umbral": int(dias_sobre),
        "total_dias": int(total_dias),
        "porcentaje_sobre_umbral": porcentaje_sobre,
    }


# ======================================================================
# Celda original 11
# ======================================================================
class VisualizadorCobertura:
    '''Agrupa los gráficos de diagnóstico usados en cada etapa del pipeline.'''

    @staticmethod
    def histograma_cobertura(
        resumen: pd.DataFrame,
        bins: Optional[list[float]] = None,
        titulo: str = "Distribución de la cobertura diaria",
    ) -> None:
        '''Histograma de días agrupados por rango de % de cobertura.'''
        bins = bins or [0, 50, 70, 80, 90, 95, 98, 99, 99.5, 99.9, 99.99, 100.01]

        categorias = pd.cut(resumen["porcentaje"], bins=bins, include_lowest=True)
        frecuencia = categorias.value_counts().sort_index()

        plt.figure(figsize=(12, 6))
        frecuencia.plot(kind="bar")
        plt.xlabel("Porcentaje de cobertura")
        plt.ylabel("Cantidad de días")
        plt.title(titulo)
        plt.xticks(rotation=45)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def dia_reconstruido(
        df: pd.DataFrame,
        fecha: pd.Timestamp,
        columna_flag: str,
        columna_valor: str = "S4",
    ) -> None:
        '''
        Grafica, para un día puntual, los datos originales vs. los puntos
        rellenados por el `ReconstructorAR` (marcados por `columna_flag`).
        '''
        datos = df[df.index.normalize() == fecha]
        originales = datos[~datos[columna_flag].astype(bool)]
        corregidos = datos[datos[columna_flag].astype(bool)]

        fig, ax = plt.subplots(figsize=(16, 5))
        ax.plot(originales.index, originales[columna_valor], ".", markersize=4, label="Datos originales")
        ax.plot(corregidos.index, corregidos[columna_valor], ".", markersize=7, label="Reconstruido (AR)")
        ax.set_title(f"{columna_valor} — {fecha:%Y-%m-%d} | {len(corregidos)} puntos completados")
        ax.set_xlabel("Hora")
        ax.set_ylabel(columna_valor)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def dias_reconstruidos(
        df: pd.DataFrame,
        columna_flag: str,
        columna_valor: str = "S4",
        max_dias: int = 20,
    ) -> None:
        '''Aplica `dia_reconstruido` a todos los días marcados en `columna_flag`.'''
        dias = df.loc[df[columna_flag].astype(bool)].index.normalize().unique()
        print(f"Días corregidos: {len(dias)}")
        for fecha in dias[:max_dias]:
            VisualizadorCobertura.dia_reconstruido(df, fecha, columna_flag, columna_valor)

    @staticmethod
    def dia_incompleto(
        df: pd.DataFrame,
        fecha: pd.Timestamp,
        info_cobertura: pd.Series,
        puntos_dia: int = PUNTOS_DIA,
        columna_valor: str = "S4",
    ) -> None:
        '''
        Grafica un día incompleto: datos reales en azul, minutos faltantes
        marcados en rojo justo debajo del mínimo observado (solo para
        visualizar *dónde* están los huecos, no su valor).
        '''
        datos_dia = df[df.index.normalize() == fecha]
        indice_completo = pd.date_range(start=fecha, periods=puntos_dia, freq="1min")
        timestamps_reales = indice_completo[indice_completo.isin(datos_dia.index)]
        timestamps_faltantes = indice_completo[~indice_completo.isin(datos_dia.index)]
        valores_reales = datos_dia.reindex(timestamps_reales)[columna_valor]

        fig, ax = plt.subplots(figsize=(16, 5))
        ax.scatter(timestamps_reales, valores_reales, s=8, color="blue", label="Datos")

        if len(valores_reales.dropna()) > 0:
            y_min, y_max = valores_reales.min(), valores_reales.max()
            y_faltante = y_min - 0.05 * (y_max - y_min) if y_max != y_min else y_min - 1
            ax.scatter(timestamps_faltantes, [y_faltante] * len(timestamps_faltantes),
                       s=12, color="red", label="Faltantes")

        ax.set_title(
            f"{columna_valor} — {fecha:%Y-%m-%d} | Cobertura: {info_cobertura['porcentaje']:.2f}% "
            f"| {info_cobertura['puntos']}/{puntos_dia} puntos"
        )
        ax.set_xlabel("Tiempo")
        ax.set_ylabel(columna_valor)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def dias_incompletos(
        df: pd.DataFrame,
        puntos_dia: int = PUNTOS_DIA,
        columna_valor: str = "S4",
        max_dias: int = 20,
    ) -> None:
        '''Aplica `dia_incompleto` a los primeros `max_dias` días con cobertura < 100 %.'''
        resumen = cobertura_diaria(df, puntos_dia)
        incompletos = resumen[resumen["porcentaje"] < 100].sort_index()
        print(f"Días incompletos a revisar: {len(incompletos)}")
        for fecha, info in incompletos.head(max_dias).iterrows():
            VisualizadorCobertura.dia_incompleto(df, fecha, info, puntos_dia, columna_valor)


# ======================================================================
# Celda original 14
# ======================================================================
# ---------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------
@dataclass
class ConfigReconstruccion:
    '''
    Parámetros de una etapa de reconstrucción de S4.

    Nuevos parámetros:
        s4_max_fisico     : techo físico de S4 aplicado SIEMPRE.
        max_gap           : longitud máxima (min) de hueco a reconstruir.
                            Huecos más largos quedan NaN. None = sin límite.
        interp_lineal_max : huecos de hasta N min con datos en ambos lados se
                            rellenan por interpolación lineal (sin AR).
        exigir_estable    : descartar ajustes AR no estacionarios.
        ventana_residuos  : el ruido bootstrap se toma solo de los residuos de
                            los últimos N minutos junto al hueco (no de todo el
                            día), para no mezclar el ruido de un evento con el
                            de un periodo quieto.
    '''
    puntos_dia: int = PUNTOS_DIA
    umbral_min: float = 0.0
    umbral_max: float = 100.0
    columna: str = "S4"
    lags: int = 20
    min_entrenamiento: int = 50
    random_seed: int = 42
    flag_col: str = "Fix"
    clip_local: bool = True
    ventana_clip: int = 60
    fallback_extremos: bool = False
    s4_max_fisico: float = S4_MAX_FISICO
    max_gap: Optional[int] = 60
    interp_lineal_max: int = 5
    exigir_estable: bool = True
    ventana_residuos: int = 120


# ---------------------------------------------------------------------
# Reconstructor
# ---------------------------------------------------------------------
class ReconstructorAR:
    '''
    Reconstrucción de huecos de S4:
        - huecos cortos      -> interpolación lineal
        - huecos medianos    -> AR bidireccional estable + bootstrap
        - huecos > max_gap   -> se dejan NaN (no se inventan datos)
    '''

    def __init__(self, config: ConfigReconstruccion):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        self.n_ar_inestables = 0

    # ---------------- selección de días ----------------
    def _dias_en_rango(self, df: pd.DataFrame) -> pd.Index:
        resumen = cobertura_diaria(df, self.config.puntos_dia, self.config.columna)
        cfg = self.config
        cond = (resumen["porcentaje"] >= cfg.umbral_min) & (resumen["porcentaje"] < cfg.umbral_max)
        return resumen.index[cond]

    # ---------------- ajuste AR estable ----------------
    def _lags_efectivos(self, n_muestras: int) -> int:
        return min(self.config.lags, max(1, n_muestras // 3))

    def _ajustar_ar_estable(self, muestra: np.ndarray, n_pred: int):
        '''
        Ajusta AutoReg y verifica estacionariedad (todas las |raíces| > 1).
        Si el ajuste es explosivo, reintenta con la mitad de lags.
        Devuelve (predicción determinística, residuos) o (None, None).
        '''
        serie = pd.Series(muestra, index=np.arange(len(muestra)))
        lags = self._lags_efectivos(len(muestra))

        while lags >= 1:
            modelo = AutoReg(serie, lags=lags, trend="c").fit()
            estable = bool(np.all(np.abs(modelo.roots) > 1.0))
            if estable or not self.config.exigir_estable:
                pred = np.asarray(
                    modelo.predict(start=len(serie), end=len(serie) + n_pred - 1), dtype=float)
                res = np.asarray(modelo.resid, dtype=float)
                res = res[np.isfinite(res)][-self.config.ventana_residuos:]
                return pred, res
            lags //= 2

        self.n_ar_inestables += 1
        return None, None

    # ---------------- reconstrucción de un hueco ----------------
    def _reconstruir_un_gap(self, serie_dia, indices_gap, serie_adelante, serie_atras, fecha):
        cfg = self.config
        n = len(indices_gap)
        hay_antes = len(serie_adelante) > 0
        hay_despues = len(serie_atras) > 0

        # 1) Huecos cortos con datos a ambos lados: interpolación lineal
        if n <= cfg.interp_lineal_max and hay_antes and hay_despues:
            a, b = serie_adelante.iloc[-1], serie_atras.iloc[0]
            pred = a + (b - a) * np.arange(1, n + 1) / (n + 1)
            return np.clip(pred, 0, cfg.s4_max_fisico), "lineal"

        # 2) AR forward / backward (solo modelos estables)
        pf = rf = pb = rb = None
        if len(serie_adelante) >= cfg.min_entrenamiento:
            try:
                pf, rf = self._ajustar_ar_estable(serie_adelante.to_numpy(), n)
            except Exception as e:
                print(f"AR FORWARD | {fecha} | gap={n} | {type(e).__name__}: {e}")
        if len(serie_atras) >= cfg.min_entrenamiento:
            try:
                pb, rb = self._ajustar_ar_estable(serie_atras.iloc[::-1].to_numpy(), n)
                if pb is not None:
                    pb = pb[::-1]
            except Exception as e:
                print(f"AR BACKWARD | {fecha} | gap={n} | {type(e).__name__}: {e}")

        # 3) Mezcla ponderada por distancia: cerca del inicio domina forward,
        #    cerca del final domina backward (sin saltos en los bordes).
        ruido = None
        if pf is not None and pb is not None:
            w = (n - np.arange(n)) / (n + 1.0)
            base = w * pf + (1 - w) * pb
            # Un único ruido bootstrap; cada punto toma el residuo del lado
            # más cercano con la misma ponderación (no mezcla ruido de evento
            # con ruido de periodo quieto).
            if len(rf) and len(rb):
                desde_forward = self.rng.random(n) < w
                ruido = np.where(desde_forward,
                                 self.rng.choice(rf, size=n, replace=True),
                                 self.rng.choice(rb, size=n, replace=True))
        elif pf is not None:
            base = pf
            ruido = self.rng.choice(rf, size=n, replace=True) if len(rf) else None
        elif pb is not None:
            base = pb
            ruido = self.rng.choice(rb, size=n, replace=True) if len(rb) else None
        else:
            base = np.full(n, np.nan)

        metodo = "ar"
        if np.isfinite(base).any():
            if ruido is not None:
                base = base + ruido
        elif cfg.fallback_extremos:
            # 4) Borde del día sin contexto suficiente para AR
            base = self._fallback_extremos(n, serie_dia, indices_gap)
            metodo = "borde"

        # 5) Restricciones: rango local y rango físico
        if cfg.clip_local:
            ctx = pd.concat([serie_adelante.tail(cfg.ventana_clip), serie_atras.head(cfg.ventana_clip)])
            if not ctx.empty:
                base = np.clip(base, ctx.min(), ctx.max())
        base = np.where(np.isfinite(base), np.clip(base, 0, cfg.s4_max_fisico), np.nan)

        return base, metodo

    @staticmethod
    def _fallback_extremos(n, serie_dia, indices_gap):
        pred = np.full(n, np.nan)
        validos = serie_dia.dropna()
        if validos.empty:
            return pred
        primer_idx, ultimo_idx = validos.index[0], validos.index[-1]
        for i, ts in enumerate(indices_gap):
            if ts < primer_idx:
                pred[i] = validos.iloc[0]
            elif ts > ultimo_idx:
                pred[i] = validos.iloc[-1]
        return pred

    # ---------------- API pública ----------------
    def reconstruir(self, df: pd.DataFrame):
        cfg = self.config
        df_resultado = df.copy()
        df_resultado[cfg.flag_col] = False
        if "Metodo" not in df_resultado.columns:
            df_resultado["Metodo"] = "original"

        dias = self._dias_en_rango(df)
        print(f"[{cfg.flag_col}] Días con cobertura en [{cfg.umbral_min}%, {cfg.umbral_max}%): {len(dias):,}")

        filas_nuevas, estadisticas = [], []
        n_omitidos, pts_omitidos = 0, 0

        for fecha in dias:
            datos_dia = df[df.index.normalize() == fecha]
            idx_full = pd.date_range(start=fecha, periods=cfg.puntos_dia, freq="1min")
            serie_dia = pd.to_numeric(datos_dia[cfg.columna], errors="coerce")
            serie_dia = serie_dia[~serie_dia.index.duplicated()].reindex(idx_full)
            faltantes = serie_dia.isna()
            if not faltantes.any():
                continue

            grupos = faltantes.ne(faltantes.shift()).cumsum()
            for _, grupo in faltantes.groupby(grupos):
                indices_gap = grupo[grupo].index
                n = len(indices_gap)
                if n == 0:
                    continue

                # Huecos demasiado largos: no se reconstruyen
                if cfg.max_gap is not None and n > cfg.max_gap:
                    n_omitidos += 1
                    pts_omitidos += n
                    estadisticas.append({"fecha": fecha, "gap_inicio": indices_gap[0],
                                         "gap_fin": indices_gap[-1], "puntos_gap": n,
                                         "puntos_reconstruidos": 0, "metodo": "omitido_largo"})
                    continue

                p0 = idx_full.get_loc(indices_gap[0])
                p1 = idx_full.get_loc(indices_gap[-1])
                antes = serie_dia.iloc[:p0].dropna().astype(float)
                despues = serie_dia.iloc[p1 + 1:].dropna().astype(float)

                pred, metodo = self._reconstruir_un_gap(serie_dia, indices_gap, antes, despues, fecha)
                ok = np.isfinite(pred)
                if ok.any():
                    nuevos = pd.DataFrame(index=indices_gap[ok], columns=df_resultado.columns)
                    nuevos[cfg.columna] = pred[ok]
                    nuevos[cfg.flag_col] = True
                    nuevos["Metodo"] = metodo
                    filas_nuevas.append(nuevos)
                estadisticas.append({"fecha": fecha, "gap_inicio": indices_gap[0],
                                     "gap_fin": indices_gap[-1], "puntos_gap": n,
                                     "puntos_reconstruidos": int(ok.sum()), "metodo": metodo})

        if filas_nuevas:
            nuevas = pd.concat(filas_nuevas)
            # Si el df de entrada es la malla (NaN ya presentes), se reemplazan esas filas
            df_resultado = df_resultado.drop(index=df_resultado.index.intersection(nuevas.index))
            df_resultado = pd.concat([df_resultado, nuevas])

        df_resultado = df_resultado.sort_index()
        df_resultado = df_resultado[~df_resultado.index.duplicated(keep="first")]
        df_resultado[cfg.flag_col] = df_resultado[cfg.flag_col].fillna(False).astype(bool)
        sin_dato = pd.to_numeric(df_resultado[cfg.columna], errors="coerce").isna()
        df_resultado.loc[sin_dato, "Metodo"] = "sin_dato"

        detalle = pd.DataFrame(estadisticas)
        self._imprimir_resumen(df, df_resultado, n_omitidos, pts_omitidos)
        return df_resultado, detalle

    def _imprimir_resumen(self, df_in, df_out, n_omitidos, pts_omitidos):
        cfg = self.config
        rec = df_out.loc[df_out[cfg.flag_col], cfg.columna].astype(float)
        print(f"\nRESULTADO ({cfg.flag_col})")
        print("=" * 70)
        print(f"Filas de entrada              : {len(df_in):,}")
        print(f"Filas incorporadas            : {len(rec):,}")
        if len(rec):
            print(f"  por método                  : {df_out.loc[df_out[cfg.flag_col], 'Metodo'].value_counts().to_dict()}")
            print(f"  rango S4 reconstruido       : [{rec.min():.3f}, {rec.max():.3f}]")
        print(f"Huecos > {cfg.max_gap} min sin reconstruir : {n_omitidos:,} ({pts_omitidos:,} minutos)")
        print(f"Ajustes AR descartados (no estacionarios): {self.n_ar_inestables:,}")
        orig = df_in[cfg.columna].dropna()
        conservados = df_out.loc[orig.index, cfg.columna].astype(float).eq(orig.astype(float)).all()
        print(f"Datos originales conservados  : {conservados}")


# ======================================================================
# Celda original 16
# ======================================================================
def filtrar_dias_por_cobertura(
    df: pd.DataFrame, umbral_min: float = 50, puntos_dia: int = PUNTOS_DIA
) -> pd.DataFrame:
    '''
    Elimina del DataFrame los días completos cuya cobertura sea menor a `umbral_min` (%).

    Se usa entre la etapa Fix1 y Fix3 para descartar días sin suficiente
    información real antes de invertir cómputo en reconstruirlos.
    '''
    resumen = cobertura_diaria(df, puntos_dia)
    dias_validos = resumen.index[resumen["porcentaje"] >= umbral_min]

    df_filtrado = df[df.index.normalize().isin(dias_validos)].copy()

    print(f"Filas antes del filtro : {len(df):,}")
    print(f"Filas después del filtro: {len(df_filtrado):,}")
    print(f"Días conservados       : {len(dias_validos):,}")
    print(f"Días eliminados (<{umbral_min}%): {(resumen['porcentaje'] < umbral_min).sum():,}")

    return df_filtrado


# ======================================================================
# Celda original 18
# ======================================================================
# =====================================================================
# Auditoría de episodios con S4 > 0.6 y corrección de picos aislados
# Se usa en dos momentos:
#   - Fix0  (sección 11.1): sobre el dato crudo, ANTES de reconstruir.
#   - 12.1  : verificación final sobre el dataset reconstruido.
# =====================================================================

@dataclass
class ConfigAuditoria:
    '''
    Criterios de la auditoría de episodios de S4 alto.

    umbral              : S4 que define un valor "fuerte" (episodio).
    nivel_intermedio    : valores entre este nivel y `umbral` se consideran
                          parte de una subida o bajada (rampa).
    tolerancia_min      : minutos por debajo del umbral que NO cortan un
                          episodio (une excedencias separadas por 1-2 min).
    ventana_rampa       : minutos antes/después del episodio donde se buscan
                          valores intermedios (subida / bajada).
    ventana_contexto    : minutos antes/después usados para medir el nivel
                          de fondo y la cobertura del contexto.
    ventana_mediana     : ancho de la mediana móvil (envolvente del evento).
    max_dur_aislado     : máximo de minutos sobre el umbral para que un
                          episodio pueda ser "pico aislado".
    min_rampa           : minutos intermedios mínimos para aceptar que hubo
                          subida (o bajada).
    max_dist_vecino     : el "vecino" de un episodio es el valor válido más
                          cercano dentro de N minutos (tolera huecos de 1-2 min
                          junto al pico, frecuentes en el dato crudo).
    corregir_borde_hueco: si True, un pico pegado a un hueco se evalúa solo con
                          el lado que tiene datos (clase "pico_borde_hueco").
    suav_evento         : si la mediana móvil supera este valor, hay una
                          envolvente sostenida (evento real).
    min_cobertura_ctx   : fracción mínima de minutos con dato en el contexto
                          para poder decidir si es aislado.
    max_frac_reconstr   : si más de esta fracción del episodio fue
                          reconstruida (Fix1/3/4), se clasifica "reconstruido".
    utc_offset_h        : desfase de hora local (Perú = -5).
    horario_tipico_lt   : (inicio, fin) en hora local del centelleo
                          post-atardecer; solo se informa, no clasifica.
    '''
    umbral: float = 0.6
    nivel_intermedio: float = 0.3
    tolerancia_min: int = 2
    ventana_rampa: int = 15
    ventana_contexto: int = 30
    ventana_mediana: int = 5
    max_dur_aislado: int = 2
    min_rampa: int = 2
    max_dist_vecino: int = 3
    corregir_borde_hueco: bool = True
    suav_evento: float = 0.4
    min_cobertura_ctx: float = 0.5
    max_frac_reconstr: float = 0.5
    utc_offset_h: int = -5
    horario_tipico_lt: tuple = (18, 4)


COLUMNAS_EPISODIO = [
    "fecha", "inicio", "fin", "hora_pico_UT", "hora_pico_LT", "S4_pico",
    "minutos_sobre_umbral", "extension_min", "S4_suav_max", "mediana_antes",
    "mediana_despues", "vecino_antes", "vecino_despues", "rampa_subida",
    "rampa_bajada", "hueco_antes", "hueco_despues", "cobertura_contexto", "frac_reconstruida", "max_sats_fuertes",
    "en_horario_tipico", "_ini", "_fin", "clase",
]

COLORES_CLASE = {
    "evento_sostenido": "tab:green",
    "pico_aislado": "tab:red",
    "pico_borde_hueco": "tab:pink",
    "dudoso": "tab:orange",
    "reconstruido": "tab:purple",
    "contexto_incompleto": "tab:gray",
}


class AuditorEventosS4:
    '''
    Detecta episodios con S4 > umbral y los clasifica en:

        evento_sostenido    : hay envolvente (subida y/o bajada gradual, o
                              mediana móvil alta): comportamiento físico.
        pico_aislado        : 1-2 min sobre el umbral, vecinos inmediatos y
                              fondo bajos, sin ningún valor intermedio antes
                              ni después: artefacto (criterio conservador).
        pico_borde_hueco    : igual que pico_aislado, pero pegado a un hueco:
                              se evalúa solo con el lado que tiene datos
                              (caída directa al fondo, sin rampa).
        reconstruido        : la mayor parte del episodio son valores
                              rellenados por el AR (no medidos).
        contexto_incompleto : no hay datos suficientes alrededor para decidir.
        dudoso              : no cumple ninguno de los criterios anteriores;
                              requiere revisión visual.
    '''

    def __init__(self, config: Optional[ConfigAuditoria] = None, columna: str = "S4"):
        self.cfg = config or ConfigAuditoria()
        self.columna = columna

    # ---------------- preparación ----------------
    def _preparar(self, df: pd.DataFrame, ultimos_n_dias: Optional[int]):
        datos = df[~df.index.duplicated(keep="first")].sort_index()
        if ultimos_n_dias:
            corte = datos.index.max().normalize() - pd.Timedelta(days=ultimos_n_dias - 1)
            datos = datos[datos.index >= corte]

        malla = pd.date_range(datos.index.min().floor("D"),
                              datos.index.max().floor("D") + pd.Timedelta("1D") - pd.Timedelta("1min"),
                              freq="1min")
        s4 = pd.to_numeric(datos[self.columna], errors="coerce").reindex(malla)

        flags = [c for c in ("Fix1", "Fix3", "Fix4") if c in datos.columns]
        if flags:
            rec = datos[flags].apply(lambda c: c.astype("boolean").fillna(False)).any(axis=1)
            rec = rec.reindex(malla, fill_value=False).astype(bool)
        else:
            rec = pd.Series(False, index=malla)

        fuertes = None
        if "N_Sats_Fuertes" in datos.columns:
            fuertes = pd.to_numeric(datos["N_Sats_Fuertes"], errors="coerce").reindex(malla)

        suav = s4.rolling(self.cfg.ventana_mediana, center=True, min_periods=3).median()
        return s4, rec, suav, fuertes

    # ---------------- episodios ----------------
    def _episodios(self, s4: pd.Series) -> list[tuple[int, int]]:
        pos = np.flatnonzero((s4 > self.cfg.umbral).to_numpy())
        if len(pos) == 0:
            return []
        cortes = np.flatnonzero(np.diff(pos) > self.cfg.tolerancia_min + 1)
        inicios = np.r_[pos[0], pos[cortes + 1]]
        fines = np.r_[pos[cortes], pos[-1]]
        return list(zip(inicios, fines))

    def _vecino(self, v: np.ndarray, pos: int, direccion: int) -> float:
        '''Primer valor válido a 1..max_dist_vecino minutos de `pos` en la dirección dada.'''
        for k in range(1, self.cfg.max_dist_vecino + 1):
            j = pos + direccion * k
            if 0 <= j < len(v) and np.isfinite(v[j]):
                return float(v[j])
        return np.nan

    def _clasificar(self, f: dict) -> str:
        c = self.cfg
        if f["frac_reconstruida"] > c.max_frac_reconstr:
            return "reconstruido"
        if (f["S4_suav_max"] >= c.suav_evento
                or (f["rampa_subida"] >= c.min_rampa and f["rampa_bajada"] >= c.min_rampa)):
            return "evento_sostenido"
        fondo_bajo = all(pd.notna(v) and v < c.nivel_intermedio
                         for v in (f["mediana_antes"], f["mediana_despues"]))
        if c.corregir_borde_hueco and f["hueco_antes"] != f["hueco_despues"]:
            # Pico en el borde de un hueco: se evalúa solo el lado con datos
            if f["hueco_antes"]:
                vecino, rampa, fondo = f["vecino_despues"], f["rampa_bajada"], f["mediana_despues"]
            else:
                vecino, rampa, fondo = f["vecino_antes"], f["rampa_subida"], f["mediana_antes"]
            if (f["minutos_sobre_umbral"] <= c.max_dur_aislado
                    and rampa == 0
                    and pd.notna(vecino) and vecino < c.nivel_intermedio
                    and pd.notna(fondo) and fondo < c.nivel_intermedio
                    and f["S4_suav_max"] < c.nivel_intermedio):
                return "pico_borde_hueco"
        if f["cobertura_contexto"] < c.min_cobertura_ctx:
            return "contexto_incompleto"
        vecinos_bajos = all(pd.notna(v) and v < c.nivel_intermedio
                            for v in (f["vecino_antes"], f["vecino_despues"]))
        if (f["minutos_sobre_umbral"] <= c.max_dur_aislado
                and f["rampa_subida"] + f["rampa_bajada"] == 0
                and vecinos_bajos
                and f["S4_suav_max"] < c.nivel_intermedio
                and fondo_bajo):
            return "pico_aislado"
        return "dudoso"

    # ---------------- API pública ----------------
    def auditar(self, df: pd.DataFrame, ultimos_n_dias: Optional[int] = None):
        '''
        Devuelve (episodios, resumen_dias).
            episodios     : una fila por episodio con sus métricas y su clase.
            resumen_dias  : una fila por día UT con episodios, por clase.
        ultimos_n_dias    : None = todo el dataset; N = solo los últimos N días.
        '''
        c = self.cfg
        s4, rec, suav, fuertes = self._preparar(df, ultimos_n_dias)
        v, r, sv = s4.to_numpy(), rec.to_numpy(), suav.to_numpy()
        idx = s4.index
        n = len(v)

        filas = []
        for ini, fin in self._episodios(s4):
            tramo = v[ini:fin + 1]
            p = ini + int(np.nanargmax(tramo))

            a0, d1 = max(0, ini - c.ventana_rampa), min(n, fin + 1 + c.ventana_rampa)
            antes_r, despues_r = v[a0:ini], v[fin + 1:d1]
            en_rampa = lambda x: int(np.sum((x >= c.nivel_intermedio) & (x <= c.umbral)))

            ca0, cd1 = max(0, ini - c.ventana_contexto), min(n, fin + 1 + c.ventana_contexto)
            ctx_antes, ctx_despues = v[ca0:ini], v[fin + 1:cd1]
            ctx = np.r_[ctx_antes, ctx_despues]

            hora_lt = (idx[p] + pd.Timedelta(hours=c.utc_offset_h)).hour
            h0, h1 = c.horario_tipico_lt
            tipico = (hora_lt >= h0 or hora_lt < h1) if h0 > h1 else (h0 <= hora_lt < h1)

            f = {
                "fecha": idx[p].normalize(),
                "inicio": idx[ini],
                "fin": idx[fin],
                "hora_pico_UT": idx[p],
                "hora_pico_LT": hora_lt,
                "S4_pico": float(v[p]),
                "minutos_sobre_umbral": int(np.nansum(tramo > c.umbral)),
                "extension_min": int(fin - ini + 1),
                "S4_suav_max": float(np.nanmax(sv[ini:fin + 1])) if np.isfinite(sv[ini:fin + 1]).any() else np.nan,
                "mediana_antes": float(np.nanmedian(ctx_antes)) if np.isfinite(ctx_antes).any() else np.nan,
                "mediana_despues": float(np.nanmedian(ctx_despues)) if np.isfinite(ctx_despues).any() else np.nan,
                "vecino_antes": self._vecino(v, ini, -1),
                "vecino_despues": self._vecino(v, fin, +1),
                "hueco_antes": not np.isfinite(self._vecino(v, ini, -1)),
                "hueco_despues": not np.isfinite(self._vecino(v, fin, +1)),
                "rampa_subida": en_rampa(antes_r),
                "rampa_bajada": en_rampa(despues_r),
                "cobertura_contexto": float(np.isfinite(ctx).mean()) if len(ctx) else 0.0,
                "frac_reconstruida": float(r[ini:fin + 1][np.isfinite(tramo)].mean()) if np.isfinite(tramo).any() else 0.0,
                "max_sats_fuertes": (float(np.nanmax(fuertes.to_numpy()[ini:fin + 1]))
                                     if fuertes is not None and np.isfinite(fuertes.to_numpy()[ini:fin + 1]).any()
                                     else np.nan),
                "en_horario_tipico": bool(tipico),
                "_ini": ini, "_fin": fin,
            }
            f["clase"] = self._clasificar(f)
            filas.append(f)

        self._s4, self._rec, self._suav = s4, rec, suav
        episodios = pd.DataFrame(filas, columns=COLUMNAS_EPISODIO)
        if episodios.empty:
            print("No hay episodios con S4 >", c.umbral)
            vacio = pd.DataFrame(columns=list(COLORES_CLASE) + ["S4_max_dia", "clase_dia"])
            vacio.index.name = "fecha"
            return episodios, vacio

        resumen_dias = (episodios.pivot_table(index="fecha", columns="clase", values="S4_pico",
                                              aggfunc="size", fill_value=0)
                        .reindex(columns=list(COLORES_CLASE), fill_value=0))
        resumen_dias["S4_max_dia"] = episodios.groupby("fecha")["S4_pico"].max()
        resumen_dias["clase_dia"] = np.select(
            [resumen_dias["evento_sostenido"] > 0,
             resumen_dias["dudoso"] + resumen_dias["contexto_incompleto"] > 0,
             resumen_dias["reconstruido"] > 0],
            ["con_evento", "revisar", "solo_reconstruido"],
            default="solo_picos_aislados")

        return episodios, resumen_dias

    # ---------------- reportes ----------------
    @staticmethod
    def imprimir_resumen(episodios: pd.DataFrame, resumen_dias: pd.DataFrame) -> None:
        if episodios.empty:
            return
        print("=" * 70)
        print("AUDITORÍA DE EPISODIOS S4 > umbral")
        print("=" * 70)
        print(f"Episodios detectados : {len(episodios):,}")
        print(f"Días con episodios   : {len(resumen_dias):,}")
        print("\nEpisodios por clase:")
        print(episodios["clase"].value_counts().to_string())
        print("\nDías por clase de día:")
        print(resumen_dias["clase_dia"].value_counts().to_string())
        fuera = episodios[~episodios["en_horario_tipico"]]
        print(f"\nEpisodios fuera del horario típico post-atardecer: {len(fuera):,}")
        if len(fuera):
            print(fuera["clase"].value_counts().to_string())

    @staticmethod
    def grafico_horario(episodios: pd.DataFrame) -> None:
        '''Hora local del pico por clase: los eventos reales deben concentrarse de noche.'''
        if episodios.empty:
            return
        fig, ax = plt.subplots(figsize=(12, 4))
        bins = np.arange(25) - 0.5
        for clase, g in episodios.groupby("clase"):
            ax.hist(g["hora_pico_LT"], bins=bins, alpha=0.6, label=f"{clase} ({len(g)})",
                    color=COLORES_CLASE.get(clase))
        ax.set_xticks(range(24))
        ax.set_xlabel("Hora local del pico")
        ax.set_ylabel("Episodios")
        ax.set_title("Hora local de los episodios con S4 > umbral, por clase")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def graficar_dia(self, episodios: pd.DataFrame, fecha) -> None:
        '''Día completo: datos medidos, reconstruidos, mediana móvil y episodios sombreados.'''
        c = self.cfg
        fecha = pd.Timestamp(fecha).normalize()
        sl = slice(fecha, fecha + pd.Timedelta("1D") - pd.Timedelta("1min"))
        s4, rec, suav = self._s4[sl], self._rec[sl], self._suav[sl]

        fig, ax = plt.subplots(figsize=(16, 5))
        ax.plot(s4[~rec].index, s4[~rec], ".", ms=4, label="Medido")
        if rec.any():
            ax.plot(s4[rec].index, s4[rec], ".", ms=6, color="tab:orange", label="Reconstruido (AR)")
        ax.plot(suav.index, suav, "-", lw=1, color="black", alpha=0.6,
                label=f"Mediana móvil {c.ventana_mediana} min")
        ax.axhline(c.umbral, color="red", ls="--", lw=1, label=f"Umbral {c.umbral}")
        ax.axhline(c.nivel_intermedio, color="gray", ls=":", lw=1, label=f"Nivel intermedio {c.nivel_intermedio}")

        eps = episodios[episodios["fecha"] == fecha]
        for _, e in eps.iterrows():
            ax.axvspan(e["inicio"] - pd.Timedelta("2min"), e["fin"] + pd.Timedelta("2min"),
                       color=COLORES_CLASE.get(e["clase"]), alpha=0.25)
            ax.annotate(e["clase"], (e["hora_pico_UT"], e["S4_pico"]), xytext=(4, 4),
                        textcoords="offset points", fontsize=8, color=COLORES_CLASE.get(e["clase"]))

        clases = ", ".join(f"{k}: {v}" for k, v in eps["clase"].value_counts().items())
        ax.set_title(f"S4 — {fecha:%Y-%m-%d} (UT) | {clases}")
        ax.set_xlabel("Hora UT")
        ax.set_ylabel("S4")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.show()

    def graficar_episodio(self, episodio: pd.Series, margen_min: int = 60) -> None:
        '''Zoom de ±margen_min alrededor de un episodio, con sus métricas en el título.'''
        c = self.cfg
        t0 = episodio["inicio"] - pd.Timedelta(minutes=margen_min)
        t1 = episodio["fin"] + pd.Timedelta(minutes=margen_min)
        s4, rec, suav = self._s4[t0:t1], self._rec[t0:t1], self._suav[t0:t1]

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(s4[~rec].index, s4[~rec], "o-", ms=3, lw=0.6, label="Medido")
        if rec.any():
            ax.plot(s4[rec].index, s4[rec], "o", ms=4, color="tab:orange", label="Reconstruido (AR)")
        ax.plot(suav.index, suav, "-", lw=1.2, color="black", alpha=0.6, label="Mediana móvil")
        ax.axhline(c.umbral, color="red", ls="--", lw=1)
        ax.axhline(c.nivel_intermedio, color="gray", ls=":", lw=1)
        ax.axvspan(episodio["inicio"], episodio["fin"] + pd.Timedelta("1min"),
                   color=COLORES_CLASE.get(episodio["clase"]), alpha=0.25)
        ax.set_title(
            f"{episodio['clase']} | pico {episodio['S4_pico']:.3f} a las {episodio['hora_pico_UT']:%Y-%m-%d %H:%M} UT "
            f"({episodio['hora_pico_LT']} h LT) | {episodio['minutos_sobre_umbral']} min > {c.umbral} | "
            f"rampa {episodio['rampa_subida']}↑/{episodio['rampa_bajada']}↓"
        )
        ax.set_xlim(t0, t1)
        ax.set_ylim(0, max(1.0, float(np.nanmax(s4.to_numpy())) * 1.1))
        ax.set_ylabel("S4")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.show()

    def graficar_episodios(self, episodios: pd.DataFrame, clases=("dudoso", "pico_aislado"),
                           max_episodios: int = 10, margen_min: int = 60) -> None:
        sel = episodios[episodios["clase"].isin(clases)]
        print(f"Episodios ({', '.join(clases)}): {len(sel)} (se muestran {min(len(sel), max_episodios)})")
        for _, e in sel.head(max_episodios).iterrows():
            self.graficar_episodio(e, margen_min)

    def graficar_dias(self, episodios, resumen_dias, clases_dia=("revisar", "solo_picos_aislados"),
                      max_dias: int = 10) -> None:
        dias = resumen_dias[resumen_dias["clase_dia"].isin(clases_dia)].index
        print(f"Días a graficar ({', '.join(clases_dia)}): {len(dias)} (se muestran {min(len(dias), max_dias)})")
        for fecha in dias[:max_dias]:
            self.graficar_dia(episodios, fecha)


# ---------------------------------------------------------------------
# Corrección de picos aislados
# ---------------------------------------------------------------------
def corregir_picos_aislados(df: pd.DataFrame, episodios: pd.DataFrame,
                            columna: str = "S4", vecinos: int = 5,
                            modo: str = "media_local", flag_col: str = "Fix0",
                            clases: tuple = ("pico_aislado", "pico_borde_hueco")):
    '''
    Corrige SOLO los episodios de las `clases` indicadas (por defecto
    "pico_aislado" y "pico_borde_hueco"). En un pico de borde de hueco el
    promedio se calcula con el único lado que tiene datos.

    modo = "media_local"  : reemplaza los minutos del pico por el PROMEDIO de
                            los valores válidos en los `vecinos` minutos antes
                            y después (el pico no entra en el promedio).
    modo = "mediana_local": igual, con la mediana.
    modo = "nan"          : deja esos minutos como hueco (NaN).

    Devuelve (df_corregido, bitacora). La bitácora guarda el valor original y
    el nuevo de cada minuto modificado. Se agrega la columna booleana `flag_col`.
    '''
    df_corr = df[~df.index.duplicated(keep="first")].sort_index().copy()
    df_corr[flag_col] = False
    s = pd.to_numeric(df_corr[columna], errors="coerce")
    bitacora = []

    for _, e in episodios[episodios["clase"].isin(clases)].iterrows():
        ini, fin = e["inicio"], e["fin"]
        rango = df_corr.index[(df_corr.index >= ini) & (df_corr.index <= fin)]
        antes = s[(s.index >= ini - pd.Timedelta(minutes=vecinos)) & (s.index < ini)]
        despues = s[(s.index > fin) & (s.index <= fin + pd.Timedelta(minutes=vecinos))]
        ref = pd.concat([antes, despues]).dropna()

        if modo == "media_local" and len(ref):
            nuevo = float(ref.mean())
        elif modo == "mediana_local" and len(ref):
            nuevo = float(ref.median())
        else:
            nuevo = np.nan

        for ts in rango:
            bitacora.append({"Tiempo": ts, "S4_original": float(s[ts]), "S4_corregido": nuevo,
                             "n_vecinos_usados": len(ref), "episodio_inicio": ini})
        df_corr.loc[rango, columna] = nuevo
        df_corr.loc[rango, flag_col] = True

    bitacora = pd.DataFrame(bitacora)
    n_ep = int(episodios["clase"].isin(clases).sum()) if len(episodios) else 0
    print(f"[{flag_col}] Picos aislados corregidos: {n_ep:,} episodios, {len(bitacora):,} minutos (modo = '{modo}')")
    if len(bitacora):
        print(f"[{flag_col}] S4 original en esos minutos : [{bitacora['S4_original'].min():.3f}, {bitacora['S4_original'].max():.3f}]")
        print(f"[{flag_col}] S4 asignado                 : [{bitacora['S4_corregido'].min():.3f}, {bitacora['S4_corregido'].max():.3f}]")
    return df_corr, bitacora


def graficar_correcciones(df_original: pd.DataFrame, df_corregido: pd.DataFrame,
                          bitacora: pd.DataFrame, columna: str = "S4",
                          margen_min: int = 60, max_casos: int = 10, umbral: float = 0.6) -> None:
    '''
    Antes / después de cada pico corregido: ventana de ±margen_min con el dato
    original, el valor eliminado (x roja) y el valor asignado (círculo verde).
    '''
    if bitacora.empty:
        print("No hay correcciones que graficar.")
        return
    casos = bitacora.groupby("episodio_inicio")
    print(f"Correcciones a graficar: {casos.ngroups} (se muestran {min(casos.ngroups, max_casos)})")
    orig = pd.to_numeric(df_original[~df_original.index.duplicated()][columna], errors="coerce").sort_index()
    corr = pd.to_numeric(df_corregido[columna], errors="coerce").sort_index()

    for k, (ini, g) in enumerate(casos):
        if k >= max_casos:
            break
        t0 = ini - pd.Timedelta(minutes=margen_min)
        t1 = g["Tiempo"].max() + pd.Timedelta(minutes=margen_min)
        fig, ax = plt.subplots(figsize=(12, 4))
        c = corr[t0:t1]
        ax.plot(c.index, c, "o-", ms=3, lw=0.6, label="Serie corregida")
        ax.plot(g["Tiempo"], g["S4_original"], "x", ms=10, mew=2, color="red", label="Valor eliminado")
        ax.plot(g["Tiempo"], g["S4_corregido"], "o", ms=8, mfc="none", mew=2, color="green",
                label="Valor asignado (promedio local)")
        ax.axhline(umbral, color="red", ls="--", lw=1)
        ax.set_xlim(t0, t1)
        ax.set_ylim(0, max(1.0, float(g["S4_original"].max()) * 1.1))
        ax.set_title(f"Pico corregido {ini:%Y-%m-%d %H:%M} UT | {g['S4_original'].max():.3f} → "
                     f"{g['S4_corregido'].iloc[0]:.3f}")
        ax.set_ylabel(columna)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.show()


# ======================================================================
# Celda original 20
# ======================================================================
# ---------------------------------------------------------------------
# Orquestador
# ---------------------------------------------------------------------
def ejecutar_pipeline_s4(df: pd.DataFrame, max_gap: Optional[int] = 60) -> dict:
    '''
    Fix1: 98-100 %  |  filtro < 50 %  |  Fix3: 90-100 %  |  Fix4: resto (+ bordes)
    En todas las etapas: AR estable, recorte local y físico, y huecos
    mayores a `max_gap` minutos quedan sin reconstruir.
    '''
    res = {"resumen_inicial": cobertura_diaria(df)}

    cfg1 = ConfigReconstruccion(umbral_min=98, umbral_max=100, flag_col="Fix1", max_gap=max_gap)
    res["df1"], res["detalle1"] = ReconstructorAR(cfg1).reconstruir(df)

    res["df2"] = filtrar_dias_por_cobertura(res["df1"], umbral_min=50)

    cfg3 = ConfigReconstruccion(umbral_min=90, umbral_max=100, flag_col="Fix3", max_gap=max_gap)
    res["df3"], res["detalle3"] = ReconstructorAR(cfg3).reconstruir(res["df2"])

    cfg4 = ConfigReconstruccion(umbral_min=0, umbral_max=100, flag_col="Fix4",
                                fallback_extremos=True, max_gap=max_gap)
    res["df4"], res["detalle4"] = ReconstructorAR(cfg4).reconstruir(res["df3"])

    for k in ("Fix0", "Fix1", "Fix3", "Fix4"):
        if k in res["df4"].columns:
            res["df4"][k] = res["df4"][k].astype("boolean").fillna(False).astype(bool)
    return res
