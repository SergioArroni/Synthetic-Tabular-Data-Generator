import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shap


def matrix_correlation(df: pd.DataFrame, name: str):
    """_summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(corr, cmap="coolwarm")

    # Agregar los números a la matriz de correlación
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            text = ax.text(
                j, i, round(corr.iloc[i, j], 2), ha="center", va="center", color="black"
            )

    fig.colorbar(cax)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.savefig(f"./img/correlation/correlation_matrix_{name}.png")
    # plt.show()f


def shap_values(df: pd.DataFrame, model):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        model (_type_): _description_

    Returns:
        _type_: _description_
    """
    explainer = shap.DeepExplainer(model, df)
    shap_values = explainer.shap_values(df)
    shap.force_plot(explainer.expected_value, shap_values, df)
    plt.savefig(f"./img/shap/shap.png")
    # plt.show()
    # return shap_values
