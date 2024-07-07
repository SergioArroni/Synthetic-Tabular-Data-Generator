import torch
import random
import numpy as np
from Hefesto.preprocess.load_data import read_data, split_data
from sklearn.ensemble import RandomForestClassifier
import shap
from sklearn.feature_selection import SequentialFeatureSelector as SFS
import matplotlib.pyplot as plt
import xgboost


def main():
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    df = read_data("data/cardio/cardio_train.csv")
    df = df.drop("id", axis=1)

    generate_data_report(df)
    n = 20000
    m = 20000
    v = 20000

    df_train, df_test, df_val = split_data(df, n, m, v)
    df_train.to_csv("data/cardio/split/cardio_train.csv", sep=";", index=False)
    df_test.to_csv("data/cardio/split/cardio_test.csv", sep=";", index=False)
    df_val.to_csv("data/cardio/split/cardio_val.csv", sep=";", index=False)

    df_small = df.sample(1000, random_state=seed)
    X = df_small.drop("cardio", axis=1)
    y = df_small["cardio"]
    # print(X)

    sfs(seed, X, y)

    shap_analysis(X, y, seed)

    generate_data_report(df_small)


def sfs(seed, X, y):
    """
    Selecciona las características más importantes usando FSS.
    """
    sfs = SFS(
        estimator=RandomForestClassifier(random_state=seed, n_jobs=-1, n_estimators=5),
        n_features_to_select="auto",
        tol=0.001,
        direction="backward",
        scoring="f1_weighted",
        cv=4,
    )
    sfs = sfs.fit(X, y)
    study = sfs.get_feature_names_out()
    with open("./prep/sfs.txt", "w") as sfs_file:
        for feature in enumerate(study):
            sfs_file.write(f"{feature}\t")


def shap_analysis(X, y, seed=42):
    """
    Realiza un análisis SHAP para explicar las predicciones del modelo.
    """
    print("Inicio de shap_analysis")
    print("Datos divididos en entrenamiento y prueba")
    bst = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)
    print("Modelo RandomForest creado")
    print("Modelo entrenado")
    explainer = shap.TreeExplainer(bst)
    shap_values = explainer.shap_values(X)
    print("Explainer SHAP creado")

    # Crear el gráfico de resumen con todas las variables
    shap.summary_plot(shap_values, X, plot_type="violin", show=False)
    plt.savefig("./prep/shap_summary.png")


def generate_data_report(df):
    """
    Genera un informe del conjunto de datos.
    """
    num_total_data = df.shape[0]
    num_bool_vars = df.select_dtypes(include=["bool"]).shape[1]
    num_categorical_vars = df.select_dtypes(include=["category", "object"]).shape[1]

    value_counts_gluc = df["gluc"].value_counts().to_dict()
    value_counts_cholesterol = df["cholesterol"].value_counts().to_dict()
    value_counts_alco = df["alco"].value_counts().to_dict()
    value_counts_smoke = df["smoke"].value_counts().to_dict()
    value_counts_active = df["active"].value_counts().to_dict()
    value_counts_cardio = df["cardio"].value_counts().to_dict()
    value_counts_gender = df["gender"].value_counts().to_dict()
    value_counts_ap_hi = df["ap_hi"].value_counts().to_dict()
    value_counts_ap_lo = df["ap_lo"].value_counts().to_dict()

    report = f"""
    Informe del Conjunto de Datos:
    ------------------------------
    Número total de datos: {num_total_data}
    Cantidad de variables booleanas: {num_bool_vars}
    Cantidad de variables categóricas: {num_categorical_vars}
    
    Valores de 'gluc':
    {value_counts_gluc}

    Valores de 'cholesterol':
    {value_counts_cholesterol}

    Valores de 'alco':
    {value_counts_alco}

    Valores de 'smoke':
    {value_counts_smoke}

    Valores de 'active':
    {value_counts_active}

    Valores de 'cardio':
    {value_counts_cardio}

    Valores de 'gender':
    {value_counts_gender}
    
    Valores de 'ap_hi':
    {value_counts_ap_hi}
    
    Valores de 'ap_lo':
    {value_counts_ap_lo}
    """
    # print(report)

    with open("./prep/data_report.txt", "w") as report_file:
        report_file.write(report)


if __name__ == "__main__":
    main()
