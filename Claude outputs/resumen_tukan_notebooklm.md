# Proyecto Tukan — Curva de rendimientos de CETES: teoría, resultados y próximos pasos

**Autor:** Julián · **Fecha:** 14 de octubre de 2026
**Propósito de este documento:** servir como fuente para que NotebookLM explique la **teoría** detrás del proyecto. El código se desarrolla por separado. Al final hay una lista de preguntas sugeridas.

**Perfil del lector:** matemático, sin formación previa en finanzas. Prefiere explicaciones con definiciones explícitas, ejemplos numéricos y sin dar nada por sabido.

---

## 1. Contexto financiero

### 1.1 ¿Qué es un CETE?

- **CETES** = Certificados de la Tesorería de la Federación. Son deuda de corto plazo del gobierno federal mexicano. Banxico los coloca mediante **subastas**.
- Son **bonos cupón cero**: no pagan intereses periódicos. Se compran **a descuento**, a un precio P menor a su **valor nominal** VN = 10 pesos, y al vencimiento se recibe VN.
- **Plazos** analizados: 28, 91, 182 y 364 días.
- Se consideran **libres de riesgo** en pesos: su tasa es el piso de cualquier otra inversión en pesos.
- La **tasa a 28 días** sigue casi uno a uno a la **tasa objetivo de Banxico**, por arbitraje.

### 1.2 Precio y tasa

Con interés simple y año comercial de 360 días:

```
P = VN / (1 + r · t/360)          ⇔          r = (VN/P − 1) · 360/t
```

- r es una **tasa nominal anual simple**: el rendimiento del periodo escalado linealmente a un año, sin capitalizar.
- **Ejemplo:** r = 7.05% a 28 días da P = 10 / (1 + 0.0705·28/360) = 9.9455. Se gana 0.0545 por título, es decir, 0.548% en 28 días.
- El **valor nominal es una convención de escala** (10 pesos para CETES, 100 para Mbonos). La tasa solo depende del cociente P/VN, así que no cambia si se cambia VN.

### 1.3 Comparar plazos: tasa efectiva

Las tasas simples de distintos plazos no son directamente comparables, porque ignoran la reinversión. La tasa efectiva anual corrige eso:

```
r_ef = (1 + r · t/360)^(360/t) − 1
```

**Ejemplo:** 7.05% a 28 días equivale a 7.28% efectiva; 7.50% a 364 días equivale a 7.50% efectiva.

### 1.4 Mbonos

Deuda del gobierno de **largo plazo** (3 a 30 años), con VN = 100 y un **cupón fijo cada 182 días**. Su precio se mueve inversamente a las tasas de mercado; su sensibilidad a esos cambios se llama **duración**. No se han modelado todavía.

---

## 2. Datos

- **Fuente:** serie mensual, de octubre de 2006 a septiembre de 2026: **240 meses × 4 plazos = 960 observaciones**, sin huecos.
- **Variable:** `VALUE_PCT` = "annualized monthly average rate", el **promedio aritmético de las tasas de las subastas del mes**, en % anual simple.
- **Columnas:** fecha, instrumento, indicador, valor y plazo (`Contract-time`).

### Notación

| Símbolo | Significado |
|---|---|
| r28ₜ = X²⁸ₜ | Tasa del CETE a 28 días en el mes t (la "función base") |
| r91ₜ, r182ₜ, r364ₜ | Tasas a 91, 182 y 364 días |
| sᵢₜ = rᵢₜ − r28ₜ | **Spread** o "distancia" del plazo i respecto a la base |
| spreadₜ = r364ₜ − r28ₜ | Pendiente de la curva usada en los modelos 1–3 |
| 𝓕ₜ | Información disponible en el mes t |

---

## 3. Lo observado en las series

Las cuatro tasas se mueven casi juntas y siguen los ciclos de política monetaria de Banxico:

| Periodo | Movimiento | Causa principal |
|---|---|---|
| 2008–2009 | Baja de ~8.5% a ~4.5% | Crisis financiera global |
| 2013–2015 | Baja hasta ~3% | Inflación baja, crecimiento débil |
| 2016–2018 | Sube a ~8.25% | Alzas de la Fed, depreciación del peso, gasolinazo |
| 2019–2021 | Baja a ~4% | Desaceleración y COVID-19 |
| 2021–2023 | Sube a ~11.25% | Inflación post-pandemia |
| 2024–2026 | Baja hacia ~6–7% | Desinflación y recortes de Banxico |

**Mecanismo:** Banxico tiene un mandato de inflación (meta de 3%) y mueve su tasa siguiendo aproximadamente una **regla de Taylor**:

```
i = r* + π + 0.5·(π − π*) + 0.5·(brecha del producto)
```

---

## 4. Hipótesis de expectativas (HE)

### 4.1 Enunciado

La tasa larga es aproximadamente el **promedio de las tasas cortas esperadas** durante su vida, más una **prima por plazo** θ:

```
r364ₜ = E[ Āₜ | 𝓕ₜ ] + θ ,        Āₜ = (1/12) Σᵢ r28ₜ₊ᵢ
```

**Intuición (arbitraje):** invertir a un año debe rendir lo mismo que renovar a 28 días 12 veces, más una compensación por inmovilizar el dinero.

### 4.2 Cómo se prueba

Se define el error de pronóstico ηₜ = Āₜ − E[Āₜ | 𝓕ₜ], que es ortogonal a 𝓕ₜ bajo expectativas racionales. Sustituyendo:

```
Āₜ − r28ₜ  =  α  +  β · (r364ₜ − r28ₜ)  +  εₜ
(cambio realizado)       (spread de hoy)
```

La HE implica **β = 1**. Esto convierte una afirmación sobre una esperanza condicional no observable en una **restricción lineal sobre un coeficiente de proyección**, que se estima por mínimos cuadrados.

**Detalles técnicos:**

- **Se usan diferencias, no niveles.** Las tasas en nivel son casi I(1) (raíz unitaria), lo que produciría regresiones espurias.
- **Errores HAC (Newey-West) con 11 rezagos.** Las ventanas de 12 meses se traslapan, así que el error es un MA(11) por construcción.

### 4.3 Resultado (promedio de 12 meses)

| | Valor |
|---|---|
| β | **1.81** (IC 95%: 1.51–2.12) |
| const | −0.79 |
| R² | 0.52 |
| Test β = 1 | z = 5.24, p ≈ 0: **se rechaza la HE pura** |
| Spread neutral | 0.79/1.81 ≈ **0.43 puntos**: la prima por plazo implícita |

**Lectura:** el spread acierta la **dirección** de los movimientos, pero los movimientos reales fueron **más grandes** de lo que el mercado anticipaba.

**Cautela:** la muestra efectiva es de unos 19 años independientes, con solo unos 5 ciclos completos.

---

## 5. Modelos OLS a 12 meses punto a punto

Variables: y = r28ₜ₊₁₂ − r28ₜ y su valor absoluto.

| | Modelo 1: \|y\| = β₀ + β₁·\|spread\| | Modelo 2: y = β₃ + β₂·spread |
|---|---|---|
| Pendiente | 1.35 (p = 0.012, IC [0.30, 2.41]) | **2.93** (p ≈ 0, IC [2.13, 3.73]) |
| Intercepto | 0.73 | −1.29 |
| R² | 0.16 | **0.43** |
| n | 228 (240 − 12 meses sin futuro observado) | 228 |

**Conclusiones:**

- El spread anticipa sobre todo **la dirección**, no la magnitud. Al quitar el signo se pierden dos tercios del poder explicativo.
- **El spread neutral vuelve a salir ≈ 0.44**, consistente con la sección 4.
- **¿Por qué β = 2.93 aquí y 1.81 antes?** Porque el cambio punto a punto a 12 meses es aproximadamente el doble del cambio promedio durante el año, si la tasa se mueve de forma gradual.
- **El AIC (806) no se interpreta en absoluto;** solo sirve para comparar modelos con la misma y. Contra un modelo sin spread, el ΔAIC es ≈ 125 a favor del spread.

---

## 6. Modelo bayesiano con coeficiente variable y volatilidad estocástica

### 6.1 Especificación

```
yₜ = α + βₜ · spreadₜ + εₜ ,         εₜ ~ N(0, σₜ²)
βₜ = β̄ + caminata aleatoria centrada (escala τ)
log σₜ² = log σ²ₜ₋₁ + νₜ ,           νₜ ~ N(0, ω²)
```

- **Parametrización no centrada:** βₜ = β̄ + τ·(Sₜ − media(S)), con Sₜ = z₁ + … + zₜ y zₜ ~ N(0, 1). Es matemáticamente el mismo modelo que escribir la caminata directamente, pero **elimina el "embudo de Neal"**, que impedía al muestreador converger.
- **Variables estandarizadas** para mejorar la geometría de la posterior.
- **Priors informados por OLS:** β̄ ~ N(β_OLS, (k·se_OLS)²). Con k = 3 es el modelo base; con k = 10, un prior casi vago para probar la sensibilidad.
- **Muestreador:** NUTS con nutpie.
- **Diagnósticos exigidos:** r̂ < 1.01, ESS > 400, 0 divergencias.

### 6.2 Resultados

| | OLS | Bayes, σ constante | Bayes, prior vago | **Bayes, σ estocástica** |
|---|---|---|---|---|
| α | −1.29 | −1.13 | −1.12 | **−0.44** |
| β̄ | 2.93 | 2.35 | 2.32 | **1.27** (IC 94% [0.84, 1.70]) |
| τ (unidades originales, por mes) | — | 0.53 | 0.53 | **0.41** |
| ω | — | — | — | **0.29** [0.23, 0.35] |

Todos convergieron (r̂ = 1.0, 0 divergencias). El prior influye menos de 1%: **los datos dominan**.

### 6.3 Comparación predictiva (LOO-PSIS)

| Modelo | elpd_loo | p_loo | Diferencia |
|---|---|---|---|
| **σ estocástica** | **−64.8** | 50.5 | — |
| σ constante | −160.4 | 21.5 | 95.6 (dse 9.95, **≈ 9.6 errores estándar**) |

Hubo advertencias de Pareto k > 0.7 en algunas observaciones influyentes. LOO es optimista con series traslapadas; la prueba más estricta sería *leave-future-out*.

### 6.4 Interpretación

1. **El spread anticipa la dirección** del movimiento de r28: β̄ > 0 con claridad.
2. **En condiciones normales, la magnitud es ≈ 1:1:** el IC de β̄ incluye 1, consistente con la HE.
3. **Spread neutral ≈ 0.44/1.27 ≈ 0.35 puntos** = prima por plazo.
4. **El β ≈ 3 de OLS era engañoso.** Lo inflaban episodios turbulentos. Con σ constante, esos meses dominan el ajuste; con σ estocástica pesan menos, como en mínimos cuadrados ponderados.
5. **La relación cambia en el tiempo (τ > 0):** βₜ > 1 en los ciclos de alzas (2015–18, 2020–22) y βₜ < 0 en algunos ciclos de recortes (2008–09, 2023–25). En los recortes, el spread positivo refleja prima por plazo y no expectativas de alzas.
6. **La volatilidad cambia (ω > 0):** σₜ identifica los periodos en los que el spread no es una señal confiable.

**Lo que no se explica:** por qué Banxico mueve las tasas (inflación, Fed, tipo de cambio). Además, los βₜ están suavizados con toda la muestra, así que **no son pronósticos en tiempo real**.

---

## 7. Siguiente etapa: el sistema de coeficientes (VAR)

### 7.1 Idea del autor

Cada tasa se construye sobre una **función base** (r28) más una **distancia** (spread). Modelar la base y las distancias permite inferir todos los plazos: rᵢ = r28 + sᵢ.

Esta es la estructura nivel + pendiente de los modelos de curva de rendimientos (por ejemplo, Nelson-Siegel).

### 7.2 Correcciones a las ecuaciones de los apuntes

- **"X²⁸ₜ = B'·spreadₜ₋₁ + …" no se deduce de "spreadₜ = B·X²⁸ₜ₋₁ + …".** Son dos relaciones distintas que pueden cumplirse a la vez. Esa retroalimentación es la razón de usar un sistema.
- **Cada ecuación necesita su propio rezago.** r28 es muy persistente, y los spreads también.

### 7.3 VAR(1)

```
Yₜ = [ r28ₜ , s91ₜ , s182ₜ , s364ₜ ]ᵀ

Yₜ = B₀ + S · Yₜ₋₁ + εₜ ,       εₜ ~ N(0, Σ)

r28ₜ  = B₀₁ + B₁₁·r28ₜ₋₁ + B₁₂·s91ₜ₋₁ + B₁₃·s182ₜ₋₁ + B₁₄·s364ₜ₋₁ + ε₁ₜ
s91ₜ  = B₀₂ + B₂₁·r28ₜ₋₁ + B₂₂·s91ₜ₋₁ + B₂₃·s182ₜ₋₁ + B₂₄·s364ₜ₋₁ + ε₂ₜ
s182ₜ = B₀₃ + B₃₁·r28ₜ₋₁ + B₃₂·s91ₜ₋₁ + B₃₃·s182ₜ₋₁ + B₃₄·s364ₜ₋₁ + ε₃ₜ
s364ₜ = B₀₄ + B₄₁·r28ₜ₋₁ + B₄₂·s91ₜ₋₁ + B₄₃·s182ₜ₋₁ + B₄₄·s364ₜ₋₁ + ε₄ₜ
```

| Bloque de S | Significado |
|---|---|
| B₁₁ | Persistencia de la tasa base |
| B₁₂, B₁₃, B₁₄ | ¿Las distancias anticipan a la base? (el hallazgo principal, ahora a 1 mes) |
| B₂₁, B₃₁, B₄₁ | ¿El nivel de la base cambia la forma de la curva? |
| Diagonal | Persistencia de cada spread |
| Resto | Transmisión entre tramos de la curva |

**S no es una matriz de correlación.** Sus entradas son coeficientes: no están acotadas, S no es simétrica y su diagonal no vale 1.

**Pronóstico:**

```
Ŷₜ₊ₕ = (I + S + … + Sʰ⁻¹)·B₀ + Sʰ·Yₜ
```

**Estabilidad:** el VAR es estable si todos los eigenvalores de S cumplen |λ| < 1.

**VECM y cointegración:** si r28 tiene raíz unitaria (λ ≈ 1), la forma adecuada es

```
ΔYₜ = c + α·βᵀ·Yₜ₋₁ + εₜ
```

Las relaciones de cointegración βᵀ para las tasas son combinaciones (1, −1), es decir, **los spreads**. El supuesto "tasa = base + distancia" equivale a afirmar que las tasas están cointegradas.

**Ventaja frente a los modelos a 12 meses:** el VAR mensual no tiene traslape, así que desaparece el problema MA(11).

**Multicolinealidad:** s91, s182 y s364 están muy correlacionados. Algunos coeficientes individuales pueden salir imprecisos o con signos que se compensan. Posibles salidas: prior de Minnesota o spreads incrementales.

### 7.4 TVP-VAR con volatilidad estocástica (el modelo completo)

Propuesto en los apuntes del autor. Es el marco de Primiceri (2005) y Cogley-Sargent.

```
Yₜ = B₀ + Bₜ · Yₜ₋₁ + εₜ

vec(Bₜ) = vec(Bₜ₋₁) + ηₜ ,         ηₜ ~ N(0, Q)            (Q diagonal: una τᵢⱼ por coeficiente)

εₜ ~ N(0, Σₜ) ,     Σₜ = Dₜ · R · Dₜ
    Dₜ = diag(σ₁ₜ, …, σ₄ₜ) ,       log σ²ᵢₜ = log σ²ᵢ,ₜ₋₁ + νᵢₜ     (volatilidad estocástica)
    R  = matriz de correlación constante,  prior LKJ(η = 2)
```

- **Las correlaciones viven en Σₜ** (choques simultáneos entre ecuaciones) **y en Q** (cambios simultáneos de coeficientes).
- **Con R no diagonal, el sistema debe estimarse conjuntamente** (verosimilitud normal multivariada), no ecuación por ecuación.
- **Tamaño:** unos 3,800 estados latentes para Bₜ más unos 960 de volatilidad. Requiere priors estrictos para τ.
- **Plan:** VAR constante → TVP con σ constante → TVP + SV + R. Se compara con LOO en cada paso.

---

## 8. Advertencias metodológicas

1. **Traslape de 12 meses** en los modelos 1–3: errores MA(11), HAC con muestra chica, bandas bayesianas algo optimistas.
2. **Pocas observaciones independientes:** unos 19 años y unos 5 ciclos de política monetaria.
3. **Coeficientes suavizados vs. tiempo real:** los βₜ usan información futura.
4. **LOO optimista con dependencia temporal;** Pareto k alto en meses turbulentos.
5. **Doble uso de datos** al tomar priors de OLS: verificado como irrelevante (< 1%).
6. **Las tasas en nivel son casi I(1):** hay que trabajar con diferencias, spreads o VECM.

---

## 9. Preguntas sugeridas para NotebookLM

1. Explica la hipótesis de expectativas de la curva de rendimientos con un ejemplo numérico, y por qué su versión pura se rechaza empíricamente en casi todos los países.
2. ¿Qué es la prima por plazo, por qué existe y por qué puede variar en el tiempo?
3. ¿Por qué β > 1 en la regresión de Campbell-Shiller se interpreta como "el mercado subestimó la magnitud"? ¿Qué relación tiene con Cov(θₜ, sₜ) < 0?
4. Explica las expectativas racionales y la ortogonalidad del error de pronóstico usando el concepto de filtración.
5. ¿Por qué una regresión entre series I(1) puede ser espuria? ¿Qué es la cointegración y por qué los spreads serían relaciones de cointegración?
6. Explica el estimador HAC de Newey-West y por qué el traslape de ventanas induce un MA(h−1).
7. ¿Qué es un modelo de coeficientes variables (state-space) y cómo se relaciona con el filtro de Kalman?
8. ¿Qué es la volatilidad estocástica? ¿En qué se diferencia de un GARCH?
9. ¿Por qué modelar la heterocedasticidad cambia el coeficiente estimado (analogía con mínimos cuadrados ponderados)?
10. Explica el "embudo de Neal" y por qué la parametrización no centrada lo resuelve.
11. ¿Qué miden r̂, ESS, las divergencias de NUTS y la validación LOO-PSIS, incluido Pareto k?
12. Explica el VAR(1), su estabilidad mediante eigenvalores, las funciones impulso-respuesta y la causalidad de Granger.
13. Explica el VECM, el teorema de representación de Granger y la prueba de Johansen.
14. Explica el TVP-VAR-SV de Primiceri (2005): por qué Σₜ se descompone en volatilidades y correlaciones, y qué es el prior LKJ.
15. ¿Qué es el prior de Minnesota y por qué ayuda con la multicolinealidad en un VAR?
16. ¿Cómo influyen las decisiones de Banxico, la Fed y el tipo de cambio en la curva de CETES? ¿Cómo se agregarían como variables exógenas?

---

## 10. Glosario breve

| Término | Definición |
|---|---|
| **Spread** | Diferencia entre la tasa de un plazo largo y la de uno corto; mide la pendiente de la curva |
| **Curva de rendimientos** | Tasas en función del plazo, en una fecha dada |
| **Prima por plazo** | Rendimiento extra exigido por prestar a plazos más largos |
| **OLS** | Mínimos cuadrados ordinarios: proyección ortogonal de y sobre el espacio columna de X |
| **HAC** | Errores estándar robustos a heterocedasticidad y autocorrelación |
| **HDI / IC** | Intervalo de mayor densidad posterior / intervalo de confianza |
| **elpd_loo** | Capacidad predictiva esperada fuera de muestra (mayor es mejor) |
| **I(1)** | Serie integrada de orden 1: su primera diferencia es estacionaria |
| **VAR / VECM** | Vector autorregresivo / vector de corrección de errores |
| **TVP** | *Time-varying parameters*: coeficientes que cambian en el tiempo |
| **SV** | Volatilidad estocástica: la log-varianza sigue su propio proceso aleatorio |

---

## 11. Resultados del sistema (VAR bayesiano con correlación R)

Estimado conjuntamente con Yₜ = [r28, s91, s182, s364]ᵀ, datos mensuales (sin traslape), R parametrizada con **Cholesky esférico** y prior LKJ(η = 2).

### 11.1 Matriz S (VAR constante + R; converge: r̂ ≤ 1.01, 0 divergencias)

```
                r28(t-1)   s91(t-1)   s182(t-1)   s364(t-1)
r28(t)           0.985      -0.398      1.238      -0.329
s91(t)           0.006       0.226      0.198       0.098
s182(t)          0.004      -0.300      0.663       0.262
s364(t)         -0.002      -0.573      0.153       0.968
```

- **Inercia de la tasa base:** B₁₁ = 0.985 (consistente con el suavizamiento de Banxico).
- **Los spreads anticipan a r28** (fila 1). Por multicolinealidad se interpreta el efecto conjunto: un empinamiento de 1 punto en toda la curva anticipa ≈ +0.51 puntos en r28 al mes siguiente. Los signos opuestos de s182 (+) y s364 (−) sugieren un papel de la curvatura.
- **El nivel de r28 no predice los spreads** (columna 1 ≈ 0): la información fluye de la forma de la curva hacia la tasa base, no al revés.
- **Persistencia de los spreads:** s364 = 0.97, s182 = 0.66, s91 = 0.23.
- **Estabilidad:** radio espectral 0.952 (vida media de un choque ≈ 14 meses).

### 11.2 Matriz R (correlación de los choques del mismo mes)

```
          r28    s91    s182   s364
r28      1.00  -0.23  -0.16  -0.19
s91     -0.23   1.00   0.89   0.71
s182    -0.16   0.89   1.00   0.86
s364    -0.19   0.71   0.86   1.00
```

- **Los spreads se sacuden en bloque (0.71–0.89):** existe un factor común de pendiente.
- **Un choque positivo a r28 aplana la curva el mismo mes** (correlaciones negativas y significativas): el tramo corto absorbe la sorpresa de política y el largo solo una parte.
- Que R no sea la identidad justifica estimar el sistema conjuntamente.

### 11.3 TVP-VAR + SV + R

- Casi todos los τᵢⱼ son compatibles con 0. Solo cambian en el tiempo la **persistencia de r28** (τ = 0.010) y la **de s364** (τ = 0.022): las relaciones cruzadas son estables a 1 mes.
- **Volatilidad estocástica concentrada en r28** (ω = 0.25); s364 = 0.12, s182 = 0.08, s91 ≈ 0.
- **LOO:** TVP-VAR + SV + R le gana al VAR constante por 55.5 (dse 12.9, ≈ 4.3 errores estándar). La ventaja proviene sobre todo de la volatilidad estocástica.
- **Convergencia incompleta** (r̂ hasta 1.03, ESS mínimo 85 en las filas de s91 y s182): conclusiones cualitativas, no decimales.

### 11.4 Lectura económica

- **Cambios graduales y anticipados de Banxico:** los spreads se mueven antes y los predicen (indicador adelantado).
- **Sorpresas de política:** no se pueden predecir, pero **su propagación sí** (funciones impulso-respuesta: R da el impacto del mes, Sʰ la propagación posterior).

---

## Anexo A. Formulación macro-financiera: regla de Taylor + hipótesis de expectativas

*Propuesta del autor, con correcciones y precisiones señaladas en A.7.*

### A.1 Tasa deseada de Banxico (economía pequeña y abierta)

```
i*ₜ = r̄* + πₜ + γ_π(πₜ − π*) + γ_y(yₜ − y*) + γ_Fed(i^Fedₜ − r^Fed*) + γ_e Δln eₜ
```

| Símbolo | Significado | Fuente mensual posible |
|---|---|---|
| r̄* | Tasa real neutral (no observable) | Se absorbe en la constante |
| πₜ − π* | Brecha de inflación (meta 3%) | INPC, variación anual (INEGI) |
| yₜ − y* | Brecha del producto (no observable) | IGAE (INEGI) menos tendencia (filtro HP u otro) |
| i^Fedₜ | Tasa de fondos federales de EE. UU. | FRED |
| Δln eₜ | Depreciación mensual del peso | Tipo de cambio FIX (Banxico) |

Agrupando constantes, con Xₜ = [1, πₜ, yₜ − y*, i^Fedₜ, Δln eₜ]ᵀ:

```
i*ₜ = γᵀ Xₜ ,        γ₀ = r̄* − γ_π π* − γ_Fed r^Fed* ,    γ₁ = 1 + γ_π   (principio de Taylor: γ₁ > 1)
```

### A.2 Suavizamiento (gradualismo)

```
r28ₜ = ρ · r28ₜ₋₁ + (1 − ρ) · γᵀXₜ + vₜ ,        vₜ ~ N(0, σ²_v)     (vₜ = sorpresa de política)
```

### A.3 Expectativas condicionales

Bajo expectativas racionales y con Eₜ[vₜ₊ⱼ] = 0 para j ≥ 1:

```
Eₜ[r28ₜ₊ₖ] = ρᵏ r28ₜ + (1 − ρ) Σⱼ₌₁ᵏ ρᵏ⁻ʲ γᵀ Eₜ[Xₜ₊ⱼ]          (k ≥ 1; por inducción)
```

### A.4 Promedio esperado a lo largo de la vida del título (M meses: 3, 6 o 12)

```
Āₜ⁽ᴹ⁾ = (1/M) Σₖ₌₀ᴹ⁻¹ Eₜ[r28ₜ₊ₖ]
      = [(1 − ρᴹ) / (M(1 − ρ))] · r28ₜ  +  (1/M) Σⱼ₌₁ᴹ⁻¹ (1 − ρᴹ⁻ʲ) · γᵀ Eₜ[Xₜ₊ⱼ]
```

La segunda forma se obtiene intercambiando el orden de la doble suma: Σₖ₌₁ᴹ⁻¹ Σⱼ₌₁ᵏ ρᵏ⁻ʲ aⱼ = Σⱼ₌₁ᴹ⁻¹ aⱼ (1 − ρᴹ⁻ʲ)/(1 − ρ). Así, cada fundamento futuro pesa wⱼ = (1 − ρᴹ⁻ʲ)/M: **los fundamentos más cercanos pesan más**.

### A.5 Descomposición del spread

```
r_M,ₜ = Āₜ⁽ᴹ⁾ + θ_M,ₜ          ⇒          s_M,ₜ = (Āₜ⁽ᴹ⁾ − r28ₜ)  +  θ_M,ₜ
                                              └─ expectativa de Banxico ─┘   └─ prima por plazo ─┘
```

**Forma compacta (caso particular).** Si se espera que los fundamentos se mantengan, Eₜ[Xₜ₊ⱼ] = Xₜ, entonces Eₜ[r28ₜ₊ₖ] = ρᵏ r28ₜ + (1 − ρᵏ) i*ₜ y:

```
s_M,ₜ = κ_M · (i*ₜ − r28ₜ) + θ_M,ₜ ,          κ_M = 1 − (1 − ρᴹ) / (M(1 − ρ))
```

**El spread es proporcional a la brecha entre la tasa deseada y la actual**, más la prima. Valores de κ_M:

| ρ (mensual) | κ₃ (91 d) | κ₆ (182 d) | κ₁₂ (364 d) |
|---|---|---|---|
| 0.985 | 0.015 | 0.037 | 0.079 |
| 0.95 | 0.049 | 0.117 | 0.234 |
| 0.90 | 0.097 | 0.219 | 0.402 |

κ crece con el plazo (los títulos largos "ven" más del ajuste futuro) y **disminuye cuando ρ → 1**.

### A.6 Integración en un VAR macro-financiero

```
Zₜ = [ πₜ , yₜ − y* , i^Fedₜ , r28ₜ , Δln eₜ , s91ₜ , s182ₜ , s364ₜ ]ᵀ ∈ ℝ⁸

Zₜ = A₀ + Φ Zₜ₋₁ + εₜ ,        εₜ ~ N(0, Σₜ) ,   Σₜ = Dₜ R Dₜ   (como en la sección 7.4)

Eₜ[Zₜ₊ⱼ] = (Σₘ₌₀ʲ⁻¹ Φᵐ) A₀ + Φʲ Zₜ
```

**Estimación en dos etapas:**

1. **VAR bayesiano de Zₜ** (extensión del código actual a k = 8) → Eₜ[Xₜ₊ⱼ] para cada mes.
2. **Regla de Taylor con suavizamiento** (regresión bayesiana de r28ₜ sobre r28ₜ₋₁ y Xₜ, con ρ ∈ (0, 1) y γ₁ > 1) → ρ̂, γ̂.
3. **Componente de expectativas** Âₜ⁽ᴹ⁾ con A.4 y **prima por plazo** como residuo: θ̂_M,ₜ = s_M,ₜ − (Âₜ⁽ᴹ⁾ − r28ₜ).
4. **Contraste:** comparar con la expectativa implícita del propio VAR, Āₜ⁽ᴹ⁾,VAR = (1/M) Σₖ e_r28ᵀ Eₜ[Zₜ₊ₖ].

### A.7 Correcciones y precisiones a la propuesta original

1. **Inercia (ρ → 1).** El factor (1 − ρᴹ)/(M(1 − ρ)) − 1 = −κ_M **tiende a 0 cuando ρ → 1** (no se vuelve "marcadamente negativo"), y es más negativo cuando ρ → 0 (límite: 1/M − 1). Lectura correcta: con mucha inercia, el ajuste hacia i* dentro de la vida del título es pequeño, así que el spread responde poco a la brecha (κ chico).
2. **"Perfectamente determinada" es una sobreafirmación.** θ̂ es un residuo y hereda todo error de especificación: variables omitidas, una regla no literal, brechas no observables (y*, r̄*) estimadas con filtros, y el supuesto de que el mercado usa la misma información que el VAR. θ̂ es **condicional al modelo**.
3. **Identificación de la regla.** La ecuación de A.2 incluye Xₜ contemporáneo; el VAR en forma reducida solo tiene rezagos, así que ρ y γ **no se recuperan de Φ** sin restricciones. Una identificación recursiva (Cholesky) con el orden de Zₜ implica: los fundamentos no reaccionan a r28 dentro del mes, r28 reacciona a los fundamentos (la regla de Taylor) y los spreads reaccionan a todo.
4. **Tipo de cambio.** Δln eₜ reacciona a las sorpresas de Banxico en el mismo mes, así que no debe ir antes que r28 en el orden recursivo. En Zₜ se colocó después de r28, y en la regla puede usarse Δln eₜ₋₁ para evitar simultaneidad.
5. **Coherencia interna.** Si la regla se estima aparte del VAR, sus pronósticos de r28 pueden diferir de los del VAR. Lo ideal es imponer la regla como la ecuación de r28 dentro de un VAR estructural; el contraste del paso 4 mide la discrepancia.
6. **Horizontes.** Con datos mensuales: M = 3 (91 días), 6 (182 días) y 12 (364 días), con k = 0, …, M − 1.
7. **Brecha del producto.** Los filtros de tendencia (HP) tienen sesgo en los extremos de la muestra: las brechas de los últimos meses son poco confiables.

### A.8 Preguntas sugeridas para NotebookLM

17. Explica la regla de Taylor con suavizamiento y el principio de Taylor (γ₁ > 1). ¿Por qué un banco central suaviza su tasa?
18. Deriva paso a paso Eₜ[r28ₜ₊ₖ] por inducción y el intercambio de la doble suma de A.4.
19. ¿Qué significa κ_M y por qué tiende a 0 cuando ρ → 1?
20. ¿Qué es la identificación recursiva (Cholesky) en un VAR estructural y qué supuestos de tiempo implica?
21. ¿Cómo se estima la brecha del producto y qué problemas tiene el filtro de Hodrick-Prescott?
22. ¿Cómo descomponen la prima por plazo los modelos macro-financieros (Ang-Piazzesi 2003, Rudebusch-Wu 2008)?
