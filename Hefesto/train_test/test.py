from sklearn.ensemble import IsolationForest
import torch


class Test:

    def __init__(self, model, df_test, df_val, seed):
        self.model = model
        self.df_test = df_test
        self.df_val = df_val
        self.seed = seed

    def evaluate_model(self):
        test_tensor = torch.tensor(self.df_test.iloc[0].values)
        x_gen = self.model(test_tensor)

        clf = IsolationForest(random_state=self.seed).fit(self.df_val.values)

        good_ele = []
        bad_ele = []

        for ele in self.df_test.values:
            if clf.predict([ele]) == 1:
                good_ele.append(ele)
            else:
                bad_ele.append(ele)

        return x_gen, good_ele, bad_ele
