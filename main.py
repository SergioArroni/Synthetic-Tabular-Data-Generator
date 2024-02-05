from Hefesto.utils.preprocess import preprocess_data, read_data, split_data
from Hefesto.utils.utils import load_model, write_results
from Hefesto.train_test.train import Train
from Hefesto.train_test.test import Test
from Hefesto.models.diffusion.diffusion import DiffusionModel


def main():
    seed = 0
    df = read_data("data\cardio\cardio_train.csv")
    df = preprocess_data(df, seed)
    train = Train(DiffusionModel)

    n = 1000
    m = 1000
    v = 1000

    df_train, df_test, df_val = split_data(df, n, m, v)

    epochs = 10
    tolerance = 0.001

    # trained_model = load_model(
    #     "./save_models/model.pt", df_train.shape[1], DiffusionModel
    # )

    train.do_train(df_train, epochs, tolerance)

    train_model = train.model

    test = Test(train_model, df_test, df_val, seed)
    x_gen, good_ele, bad_ele = test.evaluate_model()

    write_results(epochs, df_test, x_gen, good_ele, bad_ele, "./results/results.txt")


if __name__ == "__main__":
    main()
