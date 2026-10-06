"""Carga de la configuración YAML: una sola fuente de verdad para todos los experimentos."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml


def cargar_config(ruta: str | Path, **overrides) -> dict:
    """Lee el YAML y aplica overrides con notación 'seccion.clave'.

    Ejemplo: cargar_config("config/ablacion_2025_2026.yaml", **{"perdida.beta": 150.0})
    """
    with open(ruta, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for clave, valor in overrides.items():
        aplicar(cfg, clave, valor)
    return cfg


def aplicar(cfg: dict, clave: str, valor) -> dict:
    """Modifica cfg in situ: aplicar(cfg, 'perdida.beta', 150)."""
    *ruta, ultima = clave.split(".")
    nodo = cfg
    for k in ruta:
        nodo = nodo.setdefault(k, {})
    nodo[ultima] = valor
    return cfg


def variante(cfg: dict, **overrides) -> dict:
    """Copia profunda de cfg con overrides (para grids y ablación sin mutar la base)."""
    nueva = copy.deepcopy(cfg)
    for clave, valor in overrides.items():
        aplicar(nueva, clave, valor)
    return nueva


def huella(cfg: dict, secciones=("periodo", "split", "ventanas", "escalador", "perdida", "modelo", "features")) -> str:
    """Hash corto y estable de las secciones que definen un experimento.
    Dos corridas con la misma huella y la misma semilla deben dar el mismo resultado."""
    sub = {k: cfg.get(k) for k in secciones}
    txt = json.dumps(sub, sort_keys=True, default=str)
    return hashlib.sha1(txt.encode()).hexdigest()[:10]


def ruta(cfg: dict, clave: str) -> Path:
    """Resuelve una ruta de cfg['rutas'] relativa a cfg['rutas']['base']."""
    p = Path(cfg["rutas"][clave])
    return p if p.is_absolute() else Path(cfg["rutas"]["base"]) / p
