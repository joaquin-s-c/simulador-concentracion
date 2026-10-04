import numpy as np


def validar_cuotas(cuotas, tol=1e-9):
    """
    Revisa que el vector de cuotas de mercado sea válido.
    Devuelve las cuotas como arreglo de NumPy (más cómodo para calcular).

    Reglas:
    - Debe haber al menos una cuota.
    - Cada cuota debe estar entre 0 y 1.
    - Las cuotas deben sumar 1 (con una pequeña tolerancia, porque los
      decimales en computación nunca son exactos: 0.1 + 0.2 != 0.3).
    """
    s = np.asarray(cuotas, dtype=float)

    if s.ndim != 1 or s.size == 0:
        raise ValueError("Las cuotas deben ser un vector (lista) no vacío.")
    if np.any(np.isnan(s)):
        raise ValueError("Las cuotas no pueden contener valores NaN.")
    if np.any(s < 0) or np.any(s > 1):
        raise ValueError("Cada cuota debe estar entre 0 y 1.")
    if abs(s.sum() - 1) > tol:
        raise ValueError(f"Las cuotas deben sumar 1 (suman {s.sum():.6f}).")

    return s


def razon_concentracion(cuotas, k):
    """
    Ratio de concentración CRk.

    Fórmula:  CRk = s(1) + s(2) + ... + s(k)
    donde s(1) >= s(2) >= ... son las cuotas ordenadas de mayor a menor.

    Es la participación conjunta de las k empresas más grandes.
    Va de 0 a 1: cerca de 1 = mercado muy concentrado.
    """
    s = validar_cuotas(cuotas)

    if not isinstance(k, (int, np.integer)) or k < 1 or k > len(s):
        raise ValueError(f"k debe ser un entero entre 1 y {len(s)}.")

    s_ordenadas = np.sort(s)[::-1]   # orden descendente
    return float(s_ordenadas[:k].sum())


def herfindahl_hirschman(cuotas, escala_10000=False):
    """
    Índice de Herfindahl-Hirschman (IHH).

    Fórmula:  IHH = s1^2 + s2^2 + ... + sn^2
    (suma de las cuotas elevadas al cuadrado).

    Elevar al cuadrado da más peso a las empresas grandes.
    - Con cuotas en [0, 1] el IHH va de 1/n (todas iguales) a 1 (monopolio).
    - Si escala_10000=True se multiplica por 10.000, que es la escala
      que usan muchas autoridades de competencia (cuotas en porcentaje).
    """
    s = validar_cuotas(cuotas)
    ihh = float(np.sum(s ** 2))
    return ihh * 10000 if escala_10000 else ihh


def indice_entropia(cuotas, base=np.e):
    """
    Índice de entropía (IE).

    Fórmula:  IE = sum( s_i * log(1 / s_i) ) = - sum( s_i * log(s_i) )

    Interpretación (¡ojo, es al revés que el IHH!):
    - Valor alto  -> mercado poco concentrado (muchas empresas parecidas).
    - Valor bajo  -> mercado muy concentrado. IE = 0 en monopolio.
    - Máximo = log(n), cuando las n empresas tienen la misma cuota.

    Convención: 0 * log(0) se toma como 0, así que las empresas con
    cuota 0 simplemente no aportan (y evitamos calcular log(0), que da error).
    'base' es la base del logaritmo: e (natural) por defecto; usa 2 para bits.
    """
    s = validar_cuotas(cuotas)
    s = s[s > 0]                      # descartamos cuotas iguales a 0
    ie = -np.sum(s * np.log(s)) / np.log(base)
    return float(ie)


def indice_dominancia(cuotas):
    """
    Índice de dominancia (ID), versión basada en el IHH.

    Pasos:
    1) Calculamos el IHH = sum(s_i^2).
    2) Para cada empresa, su peso dentro del IHH es: h_i = s_i^2 / IHH
       (los h_i suman 1).
    3) ID = sum( h_i^2 )

    Mide qué tan "dominada" está la concentración por una sola empresa:
    - Cerca de 1: una empresa explica casi todo el IHH (dominancia fuerte).
    - Valores bajos: la concentración está repartida entre varias empresas.
    """
    s = validar_cuotas(cuotas)
    ihh = np.sum(s ** 2)
    h = (s ** 2) / ihh
    return float(np.sum(h ** 2))


# --- Ejemplo de uso ---
if __name__ == "__main__":
    cuotas = [0.40, 0.30, 0.20, 0.10]

    print("CR2 :", razon_concentracion(cuotas, k=2))   # 0.70
    print("IHH :", herfindahl_hirschman(cuotas))       # 0.30
    print("IHH (x10.000):", herfindahl_hirschman(cuotas, escala_10000=True))
    print("IE  :", indice_entropia(cuotas))
    print("ID  :", indice_dominancia(cuotas))