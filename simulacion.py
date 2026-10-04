import numpy as np
from indicadores import razon_concentracion, herfindahl_hirschman, indice_entropia, indice_dominancia
def simular_mercados(n_empresas, n_iteraciones=1000, indicador="ihh",
                     k=None, alpha=1.0, semilla=None, devolver_cuotas=False):
    """
    Simula muchos mercados al azar y calcula un indicador de concentración
    para cada uno (método de Monte Carlo).

    Parámetros
    ----------
    n_empresas : número de empresas N de cada mercado (entero >= 2).
    n_iteraciones : cuántos mercados simular (por defecto 1.000).
    indicador : "crk", "ihh", "entropia" o "dominancia".
    k : solo para "crk": cuántas empresas grandes se suman.
    alpha : parámetro de la distribución de Dirichlet (ver explicación abajo).
        alpha = 1  -> todos los repartos posibles son igual de probables.
        alpha < 1  -> mercados más concentrados (pocas empresas grandes).
        alpha > 1  -> mercados más parejos (empresas de tamaño similar).
    semilla : número para que los resultados se puedan repetir.
    devolver_cuotas : si es True, devuelve también la matriz de cuotas.

    Devuelve
    --------
    Un arreglo de largo n_iteraciones con el valor del indicador en cada
    mercado simulado (y la matriz de cuotas si devolver_cuotas=True).
    """
    # --- Validaciones de los parámetros ---
    if not isinstance(n_empresas, (int, np.integer)) or n_empresas < 2:
        raise ValueError("n_empresas debe ser un entero >= 2.")
    if not isinstance(n_iteraciones, (int, np.integer)) or n_iteraciones < 1:
        raise ValueError("n_iteraciones debe ser un entero >= 1.")
    if alpha <= 0:
        raise ValueError("alpha debe ser mayor que 0.")

    # Diccionario: nombre del indicador -> función que lo calcula.
    # (Una "lambda" es una mini-función escrita en una línea.)
    indicadores = {
        "crk": lambda s: razon_concentracion(s, k),
        "ihh": herfindahl_hirschman,
        "entropia": indice_entropia,
        "dominancia": indice_dominancia,
    }
    indicador = indicador.lower()
    if indicador not in indicadores:
        raise ValueError(f"indicador debe ser uno de: {list(indicadores)}")
    if indicador == "crk" and k is None:
        raise ValueError("Para 'crk' debes indicar k.")

    # --- Paso 1: generar todos los mercados de una vez ---
    rng = np.random.default_rng(semilla)          # generador de números al azar
    cuotas = rng.dirichlet(np.full(n_empresas, alpha), size=n_iteraciones)
    # 'cuotas' es una matriz de n_iteraciones filas y n_empresas columnas.
    # Cada fila es un mercado y suma 1.

    # --- Paso 2: calcular el indicador en cada fila ---
    calcular = indicadores[indicador]
    resultados = np.array([calcular(fila) for fila in cuotas])

    if devolver_cuotas:
        return resultados, cuotas
    return resultados


# --- Ejemplo de uso ---
if __name__ == "__main__":
    ihh = simular_mercados(n_empresas=5, n_iteraciones=1000,
                           indicador="ihh", semilla=42)
    print("IHH medio:", ihh.mean())
    print("IHH mínimo / máximo:", ihh.min(), ihh.max())

    cr3 = simular_mercados(5, indicador="crk", k=3, semilla=42)
    print("CR3 medio:", cr3.mean())