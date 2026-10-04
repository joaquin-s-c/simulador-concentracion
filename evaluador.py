"""
evaluador.py
Clasifica un valor de concentración en Baja / Moderada / Alta usando umbrales.

Todos los indicadores se trabajan en la escala de tus funciones
(cuotas entre 0 y 1): CRk y IHH van de 0 a 1, la entropía está en logaritmo
natural y la dominancia va de 1/N a 1.

OJO: no todos los umbrales tienen el mismo respaldo. Mira el campo "fuente"
de cada indicador. Si tu profesor/material de clase usa otros cortes,
cámbialos aquí (es el único lugar donde están).
"""

import numpy as np

NIVELES = ["Baja", "Moderada", "Alta"]

# "cortes" = (corte1, corte2).
# - Si "mayor_es_mas_concentrado" es True (CRk, IHH, ID):
#       valor < corte1  -> Baja ;  corte1 <= valor <= corte2 -> Moderada ;
#       valor > corte2  -> Alta
# - Si es False (entropía, donde un valor BAJO significa más concentración):
#       valor > corte1  -> Baja ;  corte2 <= valor <= corte1 -> Moderada ;
#       valor < corte2  -> Alta      (por eso corte1 > corte2)
UMBRALES = {
    "crk": {
        "cortes": (0.40, 0.60),
        "mayor_es_mas_concentrado": True,
        "tipo": "Convención de textos de Organización Industrial (definida para CR4)",
        "fuente": (
            "Regla práctica muy usada en los textos, definida para CR4: menos de 40 % "
            "= mercado competitivo, entre 40 % y 60 % = oligopolio laxo, más de 60 % "
            "= oligopolio estrecho (algunas fuentes suben el corte alto a 70 %). "
            "No es una norma legal. Si eliges un k distinto de 4, los cortes son "
            "solo una aproximación."
        ),
    },
    "ihh": {
        "cortes": (0.15, 0.25),   # = 1.500 y 2.500 en escala 0-10.000
        "mayor_es_mas_concentrado": True,
        "tipo": "Estándar de las autoridades de competencia de EE. UU. (DOJ/FTC)",
        "fuente": (
            "Guías de Fusiones Horizontales 2010 (DOJ/FTC): menos de 1.500 = no "
            "concentrado, 1.500 a 2.500 = moderadamente concentrado, más de 2.500 "
            "= altamente concentrado. Las Guías de Fusiones 2023 bajaron el corte "
            "de 'altamente concentrado' a 1.800 (y ya no definen una banda "
            "'moderada'). Para usar ese corte, cambia 0.25 por 0.18 arriba."
        ),
    },
    "entropia": {
        "cortes": (float(np.log(1 / 0.15)), float(np.log(4))),   # 1.8971 y 1.3863
        "mayor_es_mas_concentrado": False,
        "tipo": "Derivado a partir de los cortes del IHH (NO es un estándar)",
        "fuente": (
            "No conozco umbrales estándar universalmente aceptados para la entropía. "
            "Estos se derivan así: un mercado con N' empresas de igual tamaño tiene "
            "IHH = 1/N' y entropía = ln(N'). Los cortes del IHH (0,15 y 0,25) "
            "equivalen a N' = 6,67 y N' = 4 empresas iguales, y sus entropías son "
            "ln(6,67) = 1,897 y ln(4) = 1,386. Solo vale con logaritmo natural."
        ),
    },
    "dominancia": {
        "cortes": (0.25, 0.50),
        "mayor_es_mas_concentrado": True,
        "tipo": "Convención didáctica propia (NO es un estándar)",
        "fuente": (
            "No conozco umbrales estándar para este índice. Estos cortes son una "
            "convención didáctica: 1/ID se puede leer como el número equivalente "
            "de empresas dominantes. ID > 0,50 (menos de 2 empresas equivalentes) "
            "= Alta; ID entre 0,25 y 0,50 (entre 2 y 4) = Moderada; ID < 0,25 "
            "(más de 4) = Baja. Ojo: ID nunca baja de 1/N, así que con pocas "
            "empresas casi siempre saldrá Alta."
        ),
    },
}


def clasificar(codigo, valor):
    """Devuelve 'Baja', 'Moderada' o 'Alta' para el valor del indicador."""
    corte1, corte2 = UMBRALES[codigo]["cortes"]

    if UMBRALES[codigo]["mayor_es_mas_concentrado"]:
        if valor < corte1:
            return "Baja"
        if valor <= corte2:
            return "Moderada"
        return "Alta"

    # Entropía: valor alto = poca concentración
    if valor > corte1:
        return "Baja"
    if valor >= corte2:
        return "Moderada"
    return "Alta"


def formatear(codigo, valor):
    """Texto legible para un valor de cada indicador."""
    if codigo == "crk":
        return f"{valor * 100:.1f} %"
    if codigo == "ihh":
        return f"{valor * 10000:,.0f}".replace(",", ".")   # escala 0-10.000
    return f"{valor:.4f}"


def texto_regla(codigo):
    """Describe los tramos de clasificación del indicador."""
    corte1, corte2 = UMBRALES[codigo]["cortes"]
    f = lambda v: formatear(codigo, v)
    if UMBRALES[codigo]["mayor_es_mas_concentrado"]:
        return (f"Baja: < {f(corte1)} · Moderada: {f(corte1)} a {f(corte2)} "
                f"· Alta: > {f(corte2)}")
    return (f"Baja: > {f(corte1)} · Moderada: {f(corte2)} a {f(corte1)} "
            f"· Alta: < {f(corte2)}")