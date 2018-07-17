import sys
import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix,coo_matrix

from implicit.als import AlternatingLeastSquares
from implicit.approximate_als import (AnnoyAlternatingLeastSquares, FaissAlternatingLeastSquares,
                                      NMSLibAlternatingLeastSquares)
from implicit.bpr import BayesianPersonalizedRanking
from implicit.nearest_neighbours import (BM25Recommender, CosineRecommender,
                                         TFIDFRecommender, bm25_weight)


def read_dataframe(filename):
    """ Reads the original dataset csv as a pandas dataframe """

    # read in triples of user/item/confidence_weight from the input dataset
    data = pd.read_table(filename,
                             usecols=[0, 1, 2],
                             names=['user', 'item', 'weight'],
                             na_filter=False,
                             sep=',')
    data = data.sort_values('item')
    # map each item and user to a unique numeric value
    data['user'] = data['user'].astype("category")
    data['item'] = data['item'].astype("category")


    # create a sparse matrix of all the users/weight
    matrix = coo_matrix((data['weight'].astype(np.float32),
                       (data['item'].cat.codes.copy(),
                        data['user'].cat.codes.copy()))).tocsr()
    
    # get origin item for later use
    items = list(data['item'].cat.categories)
    return items,matrix


items,matrix = read_dataframe(sys.argv[1])
# items,matrix = read_dataframe("~/Downloads/sw_order_product.csv")
data = csr_matrix((matrix.data,matrix.indices,matrix.indptr))

model = AlternatingLeastSquares(factors=50)
# model = CosineRecommender()

model.fit(data)

# the item id in matrix is not equal to real raw item id
item_diff = np.ediff1d(matrix.indptr)
to_generate = sorted(np.arange(len(items)), key=lambda x: -item_diff[x])


output_filename = os.path.splitext(sys.argv[0])[0]+".csv"
with open(output_filename,"w") as f:
    for inner_item_id in to_generate:
        sim_items = model.similar_items(inner_item_id)
        for sim_item in sim_items:
            f.write("%s,%s,%s\n" % (items[inner_item_id],items[sim_item[0]],sim_item[1]) )