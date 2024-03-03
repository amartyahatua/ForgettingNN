import torch
import random
import pandas as pd


class Rank:

    def get_index(self, df_avg):
        L = []
        for val in df_avg.values.tolist():
            L.extend(val)
        x = tuple(k[1] for k in sorted((x[1], j) for j, x in enumerate(
            sorted((x, i) for i, x in enumerate(L)))))
        ord_index = [max(x) - i for i in list(x)]
        return ord_index

    def node_order(self, weights):
        average = torch.mean(weights, axis=0)
        new_average = pd.DataFrame(average.detach().numpy())
        ord_index = self.get_index(new_average)
        return ord_index

    def random(self, weights):
        random_numbers = random.sample(range(0, weights.shape[1] - 1), random.randint(0, weights.shape[1] - 1))
        ord_index = self.node_order(weights)
        for rn in random_numbers:
            try:
                ord_index[rn] = 1
            except:
                continue
        ord_index = torch.tensor(ord_index)
        return ord_index

    def top_K(self, weights, K):
        ord_index = self.node_order(weights)
        ord_index = [ind if K <= ind else 0 for ind in ord_index]
        ord_index = torch.tensor(ord_index)
        return ord_index

    def get_ranks(self, weights):
        ord_index = self.node_order(weights)
        ord_index = torch.tensor(ord_index)
        return ord_index
