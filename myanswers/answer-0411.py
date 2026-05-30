"""Predicción de popularidad de memes con RandomForest.

Pipeline de PreprocesamientoImputación: 
 - Rellena datos faltantes con la mediana vía SimpleImputer.
 - Transformación: Aplica Yeo-Johnson vía PowerTransformer para normalizar la distribución.
 - Ingeniería Cíclica: Convierte tiempo_publicacion en componentes Seno y Coseno.
 - Selección: Filtra y conserva solo el top 60% de características con SelectPercentile y f_regression.
Modelado y Evaluación
 - Algoritmo: Entrena un RandomForestRegressor con 100 estimadores y semilla fija (42).
 - Validación: Mide el error promedio de predicción mediante la métrica RMSE redondeada a 4 decimales.

Dependencias: pandas, numpy, scikit-learn.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer


def predecir_popularidad_meme(df: pd.DataFrame, target_col: str) -> dict:
    """Entrena un modelo para predecir la popularidad de un meme.

    Args:
        df (pandas.DataFrame): Dataset con las columnas 'likes', 'shares',
            'longitud_texto', 'numero_hashtags', 'tiempo_publicacion' y la
            columna objetivo indicada en ``target_col``.
        target_col (str): Nombre de la columna objetivo a predecir.

    Returns:
        dict: Diccionario con las claves:
            - 'modelo' (RandomForestRegressor): modelo entrenado.
            - 'rmse' (float): RMSE en el conjunto de prueba (escala original).
            - 'n_features_seleccionadas' (int): nº de características usadas.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()

    # 1. Imputación
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    X_imp_df = pd.DataFrame(X_imp, columns=X.columns)

    # 2. Transformación de Potencia (Yeo-Johnson)
    pt = PowerTransformer(method="yeo-johnson")
    X_pt = pt.fit_transform(X_imp)
    X_pt_df = pd.DataFrame(X_pt, columns=X.columns)

    # 3. Ingeniería de variables: Tiempo Cíclico
    hora = X_imp_df["tiempo_publicacion"].to_numpy()
    X_pt_df["hora_sin"] = np.sin(2 * np.pi * hora / 24)
    X_pt_df["hora_cos"] = np.cos(2 * np.pi * hora / 24)
    X_pt_df = X_pt_df.drop(columns=["tiempo_publicacion"])

    # 4. Selección de características (Percentil 60)
    selector = SelectPercentile(score_func=f_regression, percentile=60)
    X_sel = selector.fit_transform(X_pt_df.to_numpy(), y)
    n_features_seleccionadas = int(X_sel.shape[1])

    # 5. Modelo (Random Forest)
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_sel, y)

    # 6. Evaluación
    y_pred = modelo.predict(X_sel)
    rmse = round(float(np.sqrt(mean_squared_error(y, y_pred))), 4)

    return {
        "modelo": modelo,
        "rmse": rmse,
        "n_features_seleccionadas": n_features_seleccionadas,
    }