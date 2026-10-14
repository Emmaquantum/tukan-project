"""
Modelo macro-financiero (Anexo A) en forma independiente.

    python macro_main.py

Si ya existen idata_var_macro.nc e idata_taylor.nc (por ejemplo, creados por
index.py), los carga y tarda segundos. Si no existen, estima los modelos y los guarda.
"""
from pathlib import Path
import pandas as pd

from Data.data import data_call
from EDA.eda import Clean
from models.regresion_model import Regressors
from Data.macro import macro_call
from models.macro_finance import construir_Z, BayesianVAR, TaylorRuleBayes, TermPremium

ROOT = Path(__file__).resolve().parent

# Brecha del producto: requiere una serie de actividad actualizada (IGAE).
# Con False se omite y la muestra llega hasta donde llegan inflación, Fed y tipo de cambio.
INCLUIR_BRECHA = False


def cargar_o_ajustar(clase, Z, nombre, **kwargs_fit):
    m = clase(Z)
    path = ROOT / nombre
    if path.exists():
        print(f"\nCargando posterior guardada: {path.name}")
        m.load(path)
    else:
        print(f"\nMuestreando {path.name} con {kwargs_fit} ...")
        m.fit(**kwargs_fit)
        m.save(path)
    return m


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    # ---------- Datos de CETES ----------
    df = data_call()
    df["DATE"] = pd.to_datetime(df.DATE)
    cl = Clean(df)
    df = cl.clean()
    df = cl.separate()
    rg = Regressors(df[df.INSTRUMENT == "cetes"])

    # ---------- Datos macro y vector Z ----------
    Z = construir_Z(rg.piv, macro_call(), incluir_brecha=INCLUIR_BRECHA)
    sufijo = f"{Z.index.max():%Y%m}" + ("" if INCLUIR_BRECHA else "_sinbrecha")

    # ---------- Etapa 1: VAR macro-financiero ----------
    var = cargar_o_ajustar(BayesianVAR, Z, f"idata_var_macro_{sufijo}.nc", vol_estocastica=False)
    print("\n================ ETAPA 1: VAR MACRO-FINANCIERO ================")
    var.diagnostics()
    print("\nΦ (media posterior):")
    print(var.tabla("Phi").round(3))
    print("\nR (correlación entre choques):")
    print(var.tabla("R").round(2))

    # ---------- Etapa 2: regla de Taylor ----------
    regla = cargar_o_ajustar(TaylorRuleBayes, Z, f"idata_taylor_{sufijo}.nc")
    print("\n================ ETAPA 2: REGLA DE TAYLOR ================")
    regla.diagnostics()
    regla.plot()

    # ---------- Etapa 3: prima por plazo ----------
    print("\n================ ETAPA 3: PRIMA POR PLAZO ================")
    tp = TermPremium(var, regla, n_muestras=400).calcular()
    print(tp.resumen().round(3))
    tp.plot()

    for s in ["s91", "s182", "s364"]:
        tp.series(s).to_csv(ROOT / f"prima_plazo_{s}.csv")
    print("\nSeries guardadas en prima_plazo_s91.csv, prima_plazo_s182.csv, prima_plazo_s364.csv")


if __name__ == "__main__":
    main()
