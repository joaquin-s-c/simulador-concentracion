import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Importamos tus funciones. Si en tus archivos tienen otros nombres,
# cambia aquí los nombres.
from evaluador import NIVELES, UMBRALES, clasificar, formatear, texto_regla
from indicadores import (
    herfindahl_hirschman,
    indice_dominancia,
    indice_entropia,
    razon_concentracion,
    validar_cuotas,
)
from simulacion import simular_mercados

# ---------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------
MAX_ITERACIONES = 100_000      # límite máximo permitido (cámbialo si quieres)
ITERACIONES_DEFECTO = 1_000
TOL_SUMA_PCT = 1e-6            # tolerancia para que las cuotas sumen 100 %

# Texto que ve el usuario -> código que entiende simular_mercados
INDICADORES = {
    "CRk (ratio de concentración)": "crk",
    "IHH (Herfindahl-Hirschman)": "ihh",
    "IE (índice de entropía)": "entropia",
    "ID (índice de dominancia)": "dominancia",
}

MODO_MANUAL = "Ingresar cuotas manualmente"
MODO_AZAR = "Generar cuotas al azar"


def calcular_indicador(codigo, cuotas, k=None):
    """Calcula el indicador elegido para UN vector de cuotas (suman 1)."""
    if codigo == "crk":
        return razon_concentracion(cuotas, k)
    if codigo == "ihh":
        return herfindahl_hirschman(cuotas)
    if codigo == "entropia":
        return indice_entropia(cuotas)
    return indice_dominancia(cuotas)


def validar_porcentajes(cuotas_pct):
    """
    Revisa las cuotas ingresadas en porcentaje.
    Devuelve un texto con el error, o None si todo está bien.
    """
    if np.any(np.isnan(cuotas_pct)):
        return "Hay celdas vacías o inválidas: completa todas las cuotas."
    if np.any(cuotas_pct < 0) or np.any(cuotas_pct > 100):
        return "Cada cuota debe estar entre 0 y 100."
    suma = cuotas_pct.sum()
    if abs(suma - 100) > TOL_SUMA_PCT:
        return f"Las cuotas deben sumar 100 (ahora suman {suma:.4f})."
    return None


# ---------------------------------------------------------------
# Título y parámetros de la simulación (barra lateral)
# ---------------------------------------------------------------
st.set_page_config(page_title="Simulador de concentración de mercado")
st.title("Simulador de concentración de mercado")
st.write(
    "Genera mercados al azar (método de Monte Carlo), muestra cómo se "
    "distribuye el indicador elegido, ubica un caso particular dentro de esa "
    "distribución y evalúa si sabes clasificar su nivel de concentración."
)

st.sidebar.header("Parámetros")

nombre_indicador = st.sidebar.selectbox("Indicador", list(INDICADORES.keys()))
codigo_indicador = INDICADORES[nombre_indicador]

n_empresas = int(st.sidebar.number_input(
    "Número de empresas (N)", min_value=2, max_value=100, value=5, step=1
))

k = None
if codigo_indicador == "crk":
    k = int(st.sidebar.number_input(
        "k (empresas más grandes que se suman)",
        min_value=1, max_value=n_empresas, value=min(3, n_empresas), step=1,
    ))

n_iteraciones = int(st.sidebar.number_input(
    f"Número de iteraciones (máx. {MAX_ITERACIONES:,})".replace(",", "."),
    min_value=1, max_value=MAX_ITERACIONES, value=ITERACIONES_DEFECTO, step=1000,
))

st.warning(
    "⚠️ **Consumo de recursos:** cada iteración genera un mercado y calcula "
    "el indicador. Al aumentar el número de iteraciones (y el de empresas) "
    "la simulación usa más memoria y procesador, y **demora más** en "
    f"terminar. El máximo permitido es {MAX_ITERACIONES:,} iteraciones."
    .replace(",", ".")
)

# ---------------------------------------------------------------
# Caso particular: cuotas manuales o al azar
# ---------------------------------------------------------------
st.subheader("Caso particular")

modo = st.radio("¿Cómo quieres definir las cuotas del caso?",
                [MODO_MANUAL, MODO_AZAR], horizontal=True)

df_editado = None
if modo == MODO_MANUAL:
    st.caption("Ingresa la cuota de cada empresa en porcentaje (0 a 100). "
               "Deben sumar 100.")
    df_inicial = pd.DataFrame(
        {"Cuota (%)": [100 / n_empresas] * n_empresas},
        index=[f"Empresa {i + 1}" for i in range(n_empresas)],
    )
    # La key cambia con N: así la tabla se reinicia al cambiar el número
    # de empresas.
    df_editado = st.data_editor(
        df_inicial,
        key=f"editor_{n_empresas}",
        column_config={
            "Cuota (%)": st.column_config.NumberColumn(
                min_value=0.0, max_value=100.0, step=0.01, format="%.2f"
            )
        },
    )
    suma_actual = df_editado["Cuota (%)"].sum()
    if abs(suma_actual - 100) <= TOL_SUMA_PCT:
        st.caption(f"Suma actual: {suma_actual:.2f} % ✅")
    else:
        st.caption(f"Suma actual: {suma_actual:.2f} % (debe ser 100)")
else:
    st.caption("Al presionar **Simular** se generará un caso al azar "
               "(reparto uniforme entre todos los repartos posibles).")

# ---------------------------------------------------------------
# Simulación (solo cuando se presiona el botón)
# ---------------------------------------------------------------
# Streamlit vuelve a ejecutar TODO el archivo cada vez que el usuario toca
# algo (por ejemplo, al responder el cuestionario). Para que los resultados
# no desaparezcan, los guardamos en st.session_state.
if "n_sim" not in st.session_state:
    st.session_state["n_sim"] = 0     # cuenta cuántas simulaciones se han hecho

if st.button("Simular", type="primary"):

    # 1) Obtener y validar las cuotas del caso particular
    if modo == MODO_MANUAL:
        cuotas_pct = df_editado["Cuota (%)"].to_numpy(dtype=float)
        error = validar_porcentajes(cuotas_pct)
        if error:
            st.error(error)
            st.stop()               # corta aquí y no simula nada
        cuotas_caso = cuotas_pct / 100
        # Pequeño ajuste para que sumen exactamente 1 (evita errores de
        # redondeo en la validación de indicadores.py).
        cuotas_caso = cuotas_caso / cuotas_caso.sum()
    else:
        cuotas_caso = np.random.default_rng().dirichlet(np.ones(n_empresas))

    try:
        validar_cuotas(cuotas_caso)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    # 2) Indicador del caso particular
    valor_caso = calcular_indicador(codigo_indicador, cuotas_caso, k)

    # 3) Simulación de la distribución
    inicio = time.time()
    with st.spinner("Simulando..."):
        resultados = simular_mercados(
            n_empresas=n_empresas,
            n_iteraciones=n_iteraciones,
            indicador=codigo_indicador,
            k=k,
        )
    duracion = time.time() - inicio

    # 4) Percentil: % de mercados simulados con valor <= al del caso
    percentil = float(np.mean(resultados <= valor_caso) * 100)

    # 5) Guardar todo para mostrarlo en las siguientes ejecuciones
    st.session_state["resultado"] = {
        "parametros": (codigo_indicador, n_empresas, k, n_iteraciones),
        "codigo": codigo_indicador,
        "nombre": nombre_indicador,
        "n_empresas": n_empresas,
        "k": k,
        "n_iteraciones": n_iteraciones,
        "valor_caso": valor_caso,
        "percentil": percentil,
        "resultados": resultados,
        "cuotas_caso": cuotas_caso,
        "duracion": duracion,
    }
    st.session_state["n_sim"] += 1    # reinicia el cuestionario (ver más abajo)

# ---------------------------------------------------------------
# Mostrar resultados guardados
# ---------------------------------------------------------------
res = st.session_state.get("resultado")

if res is None:
    st.info("Elige los parámetros y presiona **Simular**.")
else:
    # Si el usuario cambió parámetros después de simular, avisamos
    if res["parametros"] != (codigo_indicador, n_empresas, k, n_iteraciones):
        st.info("Cambiaste los parámetros desde la última simulación. Lo que "
                "ves abajo corresponde a la simulación anterior; presiona "
                "**Simular** para actualizar.")

    codigo = res["codigo"]
    nombre = res["nombre"]
    valor_caso = res["valor_caso"]
    percentil = res["percentil"]
    resultados = res["resultados"]

    st.success(f"Simulación terminada en {res['duracion']:.2f} segundos.")

    # Resumen numérico
    st.subheader("Resultados")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{nombre} del caso", f"{valor_caso:.4f}")
    c2.metric("Percentil en la simulación", f"{percentil:.1f} %")
    c3.metric("Media simulada", f"{resultados.mean():.4f}")
    st.write(
        f"El **{percentil:.1f} %** de los mercados simulados tiene un valor "
        "del indicador menor o igual al de tu caso."
    )

    c4, c5, c6 = st.columns(3)
    c4.metric("Mediana simulada", f"{np.median(resultados):.4f}")
    c5.metric("Mínimo simulado", f"{resultados.min():.4f}")
    c6.metric("Máximo simulado", f"{resultados.max():.4f}")

    # Histograma con la línea vertical del caso
    fig, ax = plt.subplots()
    ax.hist(resultados, bins=30, edgecolor="black")
    ax.axvline(valor_caso, color="red", linestyle="--", linewidth=2,
               label=f"Caso particular ({valor_caso:.4f})")
    ax.set_xlabel(f"Valor del indicador: {nombre}")
    ax.set_ylabel("Frecuencia (número de mercados simulados)")
    ax.set_title(
        f"Distribución simulada ({res['n_iteraciones']} iteraciones, "
        f"{res['n_empresas']} empresas)"
    )
    ax.legend()
    st.pyplot(fig)

    with st.expander("Ver las cuotas del caso particular (%)"):
        st.dataframe(pd.DataFrame(
            {"Cuota (%)": res["cuotas_caso"] * 100},
            index=[f"Empresa {i + 1}" for i in range(res["n_empresas"])],
        ))

    # -----------------------------------------------------------
    # Evaluador
    # -----------------------------------------------------------
    st.subheader("Evalúa el caso")
    st.write(f"Según el indicador **{nombre}**, ¿qué nivel de concentración "
             "tiene el caso particular?")

    # La key incluye el número de simulación: cada simulación nueva muestra
    # el cuestionario sin respuesta marcada.
    respuesta = st.radio(
        "Tu respuesta",
        NIVELES,
        index=None,                       # empieza sin ninguna opción marcada
        horizontal=True,
        key=f"respuesta_{st.session_state['n_sim']}",
    )

    if respuesta is None:
        st.caption("Elige una opción para ver la evaluación.")
    else:
        nivel_real = clasificar(codigo, valor_caso)
        diferencia = abs(NIVELES.index(respuesta) - NIVELES.index(nivel_real))

        if diferencia == 0:
            st.success(f"✅ ¡Acertaste! El caso tiene concentración "
                       f"**{nivel_real.lower()}**.")
        elif diferencia == 1:
            st.error(f"❌ No acertaste, pero estuviste cerca (a un nivel). "
                     f"Según los umbrales, la concentración es "
                     f"**{nivel_real.lower()}**; tú respondiste "
                     f"**{respuesta.lower()}**.")
        else:
            st.error(f"❌ No acertaste (a dos niveles de distancia). Según los "
                     f"umbrales, la concentración es **{nivel_real.lower()}**; "
                     f"tú respondiste **{respuesta.lower()}**.")

        # Explicación con números
        mayor_concentrado = UMBRALES[codigo]["mayor_es_mas_concentrado"]
        if mayor_concentrado:
            frase_percentil = "igual o más concentrado"
        else:
            frase_percentil = "igual o menos concentrado"   # entropía

        st.markdown(
            f"**Por qué:**\n\n"
            f"- El valor del caso es **{formatear(codigo, valor_caso)}** "
            f"({valor_caso:.4f} en la escala de 0 a 1 de tus funciones).\n"
            f"- Los cortes de {nombre} son: {texto_regla(codigo)}.\n"
            f"- {formatear(codigo, valor_caso)} cae en el tramo "
            f"**{nivel_real}**.\n"
            f"- Percentil: el caso es {frase_percentil} que el "
            f"**{percentil:.1f} %** de los {res['n_iteraciones']} mercados "
            f"simulados de {res['n_empresas']} empresas."
        )
        st.caption(
            "Ojo: los umbrales son valores absolutos fijos, mientras que el "
            "percentil compara con mercados al azar del mismo N. Por eso "
            "pueden no coincidir: con muchas empresas, casi todos los "
            "mercados simulados son poco concentrados y un caso en el "
            "percentil 90 podría seguir siendo 'Baja' en términos absolutos."
        )
        if codigo == "crk" and res["k"] != 4:
            st.caption(f"Nota: los cortes de CRk están definidos para CR4; "
                       f"con k = {res['k']} son solo una aproximación.")

        with st.expander("¿De dónde salen los umbrales?"):
            info = UMBRALES[codigo]
            st.markdown(f"**{nombre}** — {info['tipo']}")
            st.write(info["fuente"])