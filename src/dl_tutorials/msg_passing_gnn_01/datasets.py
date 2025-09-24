"""
I am using the public `MovieLens` dataset as described in
* https://www.d2l.ai/chapter_recommender-systems/movielens.html
* https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.MovieLens.html

This dataset is tabular data of
* `user_id` - integer
* `item_id` - integer
* `rating` - integer 1-5. (Todo model needs to scale and convert to float)
* `timestamp` - integer

Additionally, I have a debugging dataset for use when coding on the commuter
rail or airline.
"""

import pandas
import lightning.pytorch as pl

# Things to understand
# * torch_geometric HeteroData
# * how to do sampling and batches


class MovieDataModule(pl.LightningDataModule):
    # ????
    pass


class DebugDataModule(pl.LigthingDataModule):
    def __init__(self):
        """Canned data for four users. Two cohorts
        * user 0 and 2: like items 3, 4, 5
        * user 1 and 3: like item 3, hate 1 and 5
        """
        self.train_data = pandas.DataFrame(
            [[0, 3, 5, 100],
             [0, 4, 4, 200],
             [1, 3, 4, 110],
             [1, 1, 1, 210],
             [2, 3, 4, 120],
             [2, 5, 5, 220],
             [3, 3, 5, 130],
             [3, 5, 1, 230],
             ],
            columns=['user_id', 'item_id', 'rating', 'timestamp'])
        self.test_data = pandas.DataFrame(
            [[0, 5, 5, 1000],
             [1, 5, 1, 1010],
             [2, 4, 5, 1020],
             [3, 1, 1, 1030],
             ],
            columns=['user_id', 'item_id', 'rating', 'timestamp'])

    # TODO: remainder of API functions
