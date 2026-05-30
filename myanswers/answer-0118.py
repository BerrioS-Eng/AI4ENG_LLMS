"""Entrenar un clasificador multietiqueta con Relevancia Binaria.

Estrategia:
 - Relevancia Binaria (Binary Relevance): un clasificador independiente por etiqueta.
 - Estimador base: SVC con kernel lineal y soporte de probabilidades.
 - Orquestación: OneVsRestClassifier entrena un SVC por cada columna de Y.

Dependencias: scikit-learn.
"""

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


def preparar_datos2(X, Y):
    """
    Entrena un clasificador multietiqueta mediante Relevancia Binaria.

    Usa un SVC lineal con probabilidades como estimador base y lo envuelve
    en un OneVsRestClassifier para ajustar un modelo por cada etiqueta.

    Parámetros
    ----------
    X : array-like o pandas.DataFrame
        Matriz de características.
    Y : array-like
        Matriz binaria de etiquetas (una columna por clase).

    Retorna
    -------
    OneVsRestClassifier
        El modelo multietiqueta completamente entrenado.
    """
    # Estimador base: SVC lineal con probabilidades y semilla fija para replicabilidad
    estimador_base = SVC(kernel='linear', probability=True)
    # Relevancia Binaria orquestada por OneVsRestClassifier
    modelo = OneVsRestClassifier(estimador_base)
    modelo.fit(X, Y)
    return modelo
