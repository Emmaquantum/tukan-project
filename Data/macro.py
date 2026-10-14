"""
Datos macroeconómicos mensuales para la regla de Taylor.

Fuentes, en orden de prioridad para cada variable:
    inpc       Banxico SIE (SP1)  →  INEGI BIE (628194)  →  FRED
    actividad  INEGI BIE (IGAE desestacionalizado, 737121)  →  FRED (termina en 2018)
    tc         Banxico SIE (FIX, SF43718, diario → promedio mensual)  →  FRED
    fedfunds   FRED (FEDFUNDS)

Las fuentes oficiales requieren un token gratuito:
    - Banxico SIE API:  https://www.banxico.org.mx/SieAPIRest/service/v1/token
    - INEGI API de Indicadores: sección "Desarrolladores" de inegi.org.mx
Los tokens se leen de variables de entorno (BANXICO_TOKEN, INEGI_TOKEN)
o de Data/tokens.json:   {"banxico": "...", "inegi": "..."}
Sin token, esa fuente se omite y se usa la siguiente.

Archivos:
    Data/macro_manual.csv  si existe, se usa tal cual (DATE,inpc,actividad,fedfunds,tc)
    Data/macro_cache.csv   caché de la última descarga (bórralo para actualizar)
"""
import os
import json
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
ARCHIVO_MANUAL = RAIZ / "macro_manual.csv"
ARCHIVO_CACHE = RAIZ / "macro_cache.csv"
ARCHIVO_TOKENS = RAIZ / "tokens.json"
COLUMNAS = ["inpc", "actividad", "fedfunds", "tc"]
INICIO = "1990-01-01"


# ======================================================================
# Tokens
# ======================================================================
def _token(nombre):
    env = os.environ.get(f"{nombre.upper()}_TOKEN")
    if env:
        return env.strip()
    if ARCHIVO_TOKENS.exists():
        return json.loads(ARCHIVO_TOKENS.read_text(encoding="utf-8")).get(nombre, "").strip() or None
    return None


def _get_json(url, headers=None):
    req = urllib.request.Request(url, headers={"User-Agent": "tukan-project", **(headers or {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def _mensual(s):
    """Índice a inicio de mes y promedio mensual (para series diarias)."""
    s = s.dropna()
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp()
    return s.groupby(level=0).mean()


# ======================================================================
# Fuentes
# ======================================================================
def parse_banxico(js):
    datos = js["bmx"]["series"][0]["datos"]
    fechas = pd.to_datetime([d["fecha"] for d in datos], format="%d/%m/%Y")
    valores = pd.to_numeric([d["dato"].replace(",", "") for d in datos], errors="coerce")  # "N/E" → NaN
    return pd.Series(valores, index=fechas)


def banxico(serie):
    token = _token("banxico")
    if not token:
        raise RuntimeError("sin token de Banxico")
    url = (f"https://www.banxico.org.mx/SieAPIRest/service/v1/series/{serie}"
           f"/datos/{INICIO}/{date.today():%Y-%m-%d}")
    return _mensual(parse_banxico(_get_json(url, {"Bmx-Token": token})))


def parse_inegi(js):
    obs = js["Series"][0]["OBSERVATIONS"]
    fechas = pd.to_datetime([o["TIME_PERIOD"].replace("/", "-") + "-01" for o in obs])
    valores = pd.to_numeric([o["OBS_VALUE"] for o in obs], errors="coerce")
    return pd.Series(valores, index=fechas).sort_index()


def inegi(indicador):
    token = _token("inegi")
    if not token:
        raise RuntimeError("sin token de INEGI")
    ultimo_error = None
    for fuente in ("BIE", "BISE"):           # el IGAE vive en el BIE; se prueba BISE por si acaso
        for area in ("00", "0700"):          # nacional: la guía oficial usa 00; algunos ejemplos usan 0700
            url = ("https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml/INDICATOR/"
                   f"{indicador}/es/{area}/false/{fuente}/2.0/{token}?type=json")
            try:
                js = _get_json(url)
                if isinstance(js, list):     # INEGI responde una lista de textos cuando hay error
                    raise ValueError(" | ".join(map(str, js)))
                return _mensual(parse_inegi(js))
            except Exception as e:
                ultimo_error = e
    raise ultimo_error


def fred(serie):
    df = pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={serie}")
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(df.iloc[:, 0])
    return _mensual(s)


# Para cada variable: lista de (etiqueta, función, identificador), en orden de prioridad.
FUENTES = {
    "inpc": [
        ("Banxico SP1", banxico, "SP1"),
        ("INEGI 628194", inegi, "628194"),
        ("FRED MEXCPIALLMINMEI", fred, "MEXCPIALLMINMEI"),
    ],
    "actividad": [
        ("INEGI IGAE 737121", inegi, "737121"),
        ("FRED MEXPRINTO01IXOBSAM", fred, "MEXPRINTO01IXOBSAM"),
    ],
    "fedfunds": [
        ("FRED FEDFUNDS", fred, "FEDFUNDS"),
    ],
    "tc": [
        ("Banxico FIX SF43718", banxico, "SF43718"),
        ("FRED EXMXUS", fred, "EXMXUS"),
    ],
}


def _descargar_variable(nombre):
    errores = []
    for etiqueta, func, ident in FUENTES[nombre]:
        try:
            s = func(ident)
            if len(s) == 0:
                raise ValueError("serie vacía")
            print(f"  ✓ {nombre:10s} ← {etiqueta:24s} {s.index.min():%Y-%m} → {s.index.max():%Y-%m}"
                  f"   último valor: {s.iloc[-1]:,.3f}")
            return s
        except Exception as e:
            errores.append(f"{etiqueta}: {e}")
            print(f"  ✗ {nombre:10s} ← {etiqueta:24s} ({e})")
    raise RuntimeError(f"No se pudo obtener '{nombre}'. Intentos:\n  " + "\n  ".join(errores)
                       + f"\nSolución: agrega tokens (ver Data/macro.py) o crea {ARCHIVO_MANUAL.name}.")


# ======================================================================
# Función principal
# ======================================================================
def _normalizar(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()
    return df.groupby(level=0).mean()[COLUMNAS].astype(float)


def macro_call(refrescar=False, verbose=True):
    if ARCHIVO_MANUAL.exists():
        fuente = ARCHIVO_MANUAL.name
        df = pd.read_csv(ARCHIVO_MANUAL, index_col=0, parse_dates=True)
    elif ARCHIVO_CACHE.exists() and not refrescar:
        fuente = ARCHIVO_CACHE.name
        df = pd.read_csv(ARCHIVO_CACHE, index_col=0, parse_dates=True)
    else:
        fuente = "descarga"
        print("\nDescargando datos macro:")
        df = pd.DataFrame({c: _descargar_variable(c) for c in COLUMNAS})
        df.index.name = "DATE"
        df.to_csv(ARCHIVO_CACHE)

    df = _normalizar(df)

    if verbose:
        print(f"\nDatos macro desde: {fuente}")
        fines = {}
        for col in df.columns:
            s = df[col].dropna()
            fines[col] = s.index.max()
            print(f"  {col:10s} {s.index.min():%Y-%m} → {s.index.max():%Y-%m}  ({len(s)} meses)")
        fin_max = max(fines.values())
        atrasadas = [c for c, f in fines.items() if (fin_max - f).days > 183]
        if atrasadas:
            print(f"  ⚠ Terminan más de 6 meses antes que el resto: {atrasadas}. "
                  f"La muestra se recortará; considera fuentes oficiales o macro_manual.csv.")
    return df
