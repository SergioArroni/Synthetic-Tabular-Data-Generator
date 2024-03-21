import torch

from Hefesto.preprocess.load_data import read_data, split_data
from Hefesto.preprocess.correlations import matrix_correlation
from Hefesto.utils.utils import plot_statistics
from Hefesto.preprocess.preprocess import Preprocess

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    seed = 42
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    df = read_data("data/cardio/cardio_train.csv")
    df = df.drop("id", axis=1)
    plot_statistics(df, f"./img/stadistics/cardio/bruto/boxplot")
    matrix_correlation(df, "all")

    # prep = Preprocess(df)
    # prep.scaler_method()
    # df = prep.df

    # plot_statistics(df, f"./img/stadistics/cardio/standar/boxplot")

    # df["new_column"] = 0

    n = 5000
    m = 5000
    v = 5000

    df_train, df_test, df_val = split_data(df, n, m, v)
    df_train.to_csv("data/cardio/split/cardio_train.csv", sep=";", index=False)
    df_test.to_csv("data/cardio/split/cardio_test.csv", sep=";", index=False)
    df_val.to_csv("data/cardio/split/cardio_val.csv", sep=";", index=False)
    
    matrix_correlation(df, "val")

    X = df_val.drop("cardio", axis=1)
    y = df_val["cardio"]
    
    metrics = evaluate_regression(seed, X, y)
    
    a = open("results/metrics.txt", "w")
    a.write(f"F1: {metrics[0]}\n")
    a.write(f"Accuracy: {metrics[1]}\n")
    a.close()


def evaluate_regression(seed, X, y):
    """
    Evalúa el desempeño de un modelo de regresión sobre los datos generados.
    """
    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    # Escalar los datos
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Entrenar el modelo
    model = RandomForestClassifier(random_state=seed)
    model.fit(X_train, y_train)

    # Evaluar el modelo
    predictions = model.predict(X_test)

    f1 = f1_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)

    return (f1, accuracy)


if __name__ == "__main__":
    main()
