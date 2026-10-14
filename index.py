import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importar funciones especiales
from Data.data import data_call
from EDA.eda import Clean
from models.regresion_model import Regressors
from models.bayesian_model import BayesianTVP
from models.bayes_var import BayesianTVPVAR
from Data.macro import macro_call
from models.macro_finance import construir_Z, BayesianVAR, TaylorRuleBayes, TermPremium
import arviz as az


def main():
    # ---------- DATA ----------
    df = data_call()
    df["DATE"] = pd.to_datetime(df.DATE)

    cl = Clean(df)
    df = cl.clean()
    df = cl.separate()

    # ---------- CETES ----------
    df_cetes = df[df.INSTRUMENT == "cetes"]
    print("CETES!")
    df_cetes = df[df.INSTRUMENT == "cetes"]
    #print(df_cetes)
    print(df_cetes.info())
    cl_cetes = Clean(df_cetes)
    cetes_unique = cl_cetes.uniques()
    unique_cetes_df = pd.DataFrame.from_dict(cetes_unique, orient="index", columns=["n_unicos"]).T
    print(unique_cetes_df)

    #DATA
    dfs_cetes = cl_cetes.discriminator()

    plt.figure(figsize=(11, 5))
    for plazo, d in dfs_cetes.items():
        if (plazo == 28) | (plazo == 364):
            cl_cetes.plot_serietimes(d, label=f"{plazo} días")
    plt.title("CETES – tasa anualizada por plazo")
    plt.xlabel("Fecha")
    plt.ylabel("Tasa (%)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

    # ---------- MODELOS OLS ----------
    rg = Regressors(df_cetes).fit_model()
    rg.comparative_table()
    rg.graficar_model()

    # ---------- MODELO BAYESIANO ----------
    bayes = BayesianTVP.from_regressors(rg)
    ROOT = Path(__file__).resolve().parent

    def cargar_o_ajustar(nombre, **kwargs_fit):
        b = BayesianTVP.from_regressors(rg)
        path = ROOT / nombre
        if path.exists():
            print(f"\nCargando posterior guardada: {path.name}")
            b.load(path)
        else:
            print(f"\nMuestreando {path.name} con {kwargs_fit} ...")
            b.fit(**kwargs_fit)
            b.save(path)
        return b

    bayes      = cargar_o_ajustar("idata_tvp_base.nc", inflacion=3,  vol_estocastica=False)
    bayes_vago = cargar_o_ajustar("idata_tvp_vago.nc", inflacion=10, vol_estocastica=False)
    bayes_sv   = cargar_o_ajustar("idata_tvp_sv.nc",   inflacion=3,  vol_estocastica=True)

    # ---------- DIAGNÓSTICOS ----------
    for nombre, b in [("BASE", bayes), ("PRIOR VAGO", bayes_vago), ("VOLATILIDAD ESTOCÁSTICA", bayes_sv)]:
        print(f"\n================ DIAGNÓSTICO: {nombre} ================")
        b.diagnostics()
        tau_orig = b.idata.posterior["tau"] * b.y_sd / b.x_sd
        print(f"τ en unidades originales: {float(tau_orig.mean()):.3f} por mes")

    # ---------- GRÁFICAS ----------
    bayes.plot()        # σ constante
    bayes_sv.plot()     # σ variable en el tiempo

    # ---------- TABLA COMPARATIVA ----------
    def resumen(b):
        p = b.idata.posterior
        return {
            "α":        float(p["alpha"].mean()),
            "β̄":        float(p["beta_bar"].mean()),
            "τ (orig)": float((p["tau"] * b.y_sd / b.x_sd).mean()),
        }

    comp = pd.DataFrame({
        "OLS":  {"α": rg.m2.params["const"], "β̄": rg.m2.params["spread"], "τ (orig)": np.nan},
        "Base": resumen(bayes),
        "Vago": resumen(bayes_vago),
        "SV":   resumen(bayes_sv),
    })
    comp["dif % prior (vago vs base)"] = 100 * (comp["Vago"] / comp["Base"] - 1)

    print("\n================ RESUMEN DE RESULTADOS ================")
    print(comp.round(3))

    print("\n================ COMPARACIÓN LOO ================")
    print(az.compare({"σ constante": bayes.idata, "σ estocástica": bayes_sv.idata}))


    # ==================== SISTEMA: TVP-VAR (4 ecuaciones + correlación R) ====================
    #   Y_t = B0 + B_t · Y_{t-1} + ε_t ,   ε_t ~ N(0, D_t R D_t),   Y = [r28, s91, s182, s364]
    def cargar_o_ajustar_var(nombre, **kwargs_fit):
        m = BayesianTVPVAR.from_regressors(rg)
        path = ROOT / nombre
        if path.exists():
            print(f"\nCargando posterior guardada: {path.name}")
            m.load(path)
        else:
            print(f"\nMuestreando {path.name} con {kwargs_fit} ...")
            m.fit(**kwargs_fit)
            m.save(path)
        return m

    # Paso 1: VAR con S constante + correlación R  (rápido, < 1 min)
    var_const = cargar_o_ajustar_var("idata_var_const.nc",
                                     tvp=False, vol_estocastica=False, correlacion=True)
    # Paso 2: modelo completo, B_t variable + volatilidad estocástica + R  (varios minutos)
    var_full  = cargar_o_ajustar_var("idata_tvpvar_full.nc",
                                     tvp=True, vol_estocastica=True, correlacion=True)

    for nombre, m in [("VAR CONSTANTE + R", var_const), ("TVP-VAR + SV + R", var_full)]:
        print(f"\n================ DIAGNÓSTICO: {nombre} ================")
        m.diagnostics()
        print("\nB̄ (matriz S promedio):")
        print(m.tabla("Bbar").round(3))
        radio = np.abs(np.linalg.eigvals(m.tabla("Bbar").values)).max()
        print(f"Radio espectral de B̄: {radio:.4f}  ({'estable' if radio < 1 else 'NO estable'})")
        print("\nR (correlación entre choques):")
        print(m.tabla("R").round(2))
        print(m.tabla_R_intervalos().round(2))

    var_full.plot_B_t()
    var_full.plot_sigma()

    print("\n================ COMPARACIÓN LOO: SISTEMA ================")
    print(az.compare({"VAR constante + R": var_const.idata, "TVP-VAR + SV + R": var_full.idata}))



    # ==================== MODELO MACRO-FINANCIERO (Anexo A) ====================
    #   Etapa 1: VAR de Z = [π, brecha, fed, r28, Δln e, s91, s182, s364]  →  E_t[Z_{t+j}]
    #   Etapa 2: regla de Taylor con suavizamiento                         →  ρ, γ
    #   Etapa 3: spread = expectativa de Banxico + prima por plazo θ
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    macro = macro_call()
    INCLUIR_BRECHA = False   # True solo si hay serie de actividad (IGAE) actualizada
    Z = construir_Z(rg.piv, macro, incluir_brecha=INCLUIR_BRECHA)
    sufijo = f"{Z.index.max():%Y%m}" + ("" if INCLUIR_BRECHA else "_sinbrecha")

    def cargar_o_ajustar_mf(clase, nombre, **kwargs_fit):
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

    var_macro = cargar_o_ajustar_mf(BayesianVAR, f"idata_var_macro_{sufijo}.nc", vol_estocastica=False)
    regla     = cargar_o_ajustar_mf(TaylorRuleBayes, f"idata_taylor_{sufijo}.nc")

    print("\n================ ETAPA 1: VAR MACRO-FINANCIERO ================")
    var_macro.diagnostics()
    print("\nΦ (media posterior):")
    print(var_macro.tabla("Phi").round(3))
    print("\nR (correlación entre choques):")
    print(var_macro.tabla("R").round(2))

    print("\n================ ETAPA 2: REGLA DE TAYLOR ================")
    regla.diagnostics()
    regla.plot()

    print("\n================ ETAPA 3: PRIMA POR PLAZO ================")
    tp = TermPremium(var_macro, regla, n_muestras=400).calcular()
    print(tp.resumen().round(3))
    tp.plot()

    # series completas por plazo, para análisis posterior
    for s in ["s91", "s182", "s364"]:
        tp.series(s).to_csv(ROOT / f"prima_plazo_{s}.csv")
    print("\nSeries guardadas en prima_plazo_s91.csv, prima_plazo_s182.csv, prima_plazo_s364.csv")


if __name__ == "__main__":
    main()