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

class ImplicitRecommend:

    def __init__(self):
        self.model = None
        self.inner_id_to_item_ids = None
        self.item_id_to_inner_ids = None

    def fit(self,df_source: pd.DataFrame):
        data = df_source.copy()
        data['user'] = data['user'].astype("category")
        data['item'] = data['item'].astype("category")


        # create a sparse matrix of all the users/weight
        matrix = coo_matrix((data['weight'].astype(np.float32),
                        (data['item'].cat.codes.copy(),
                            data['user'].cat.codes.copy()))).tocsr()
        
        # get origin item for later use
        items = list(data['item'].cat.categories)
        data = csr_matrix((matrix.data,matrix.indices,matrix.indptr))
        self.model = AlternatingLeastSquares(factors=50)
        self.model.fit(data)
        # 这里有点不明白作者当初为什么这么写
        # item_diff = np.ediff1d(matrix.indptr)
        # self.inner_id_to_item_ids = sorted(np.arange(len(items)), key=lambda x: -item_diff[x])
        self.inner_id_to_item_ids = items
        self.item_id_to_inner_ids = {v: k for k,v in enumerate(items)}
        

    def get_similar_items(self,item_id,count: int=10):
        inner_item_id = self.item_id_to_inner_ids[item_id]
        sim_item_inner_ids = self.model.similar_items(inner_item_id,count+1)
        #return [(self.inner_id_to_item_ids[inner_id],weight) for inner_id,weight in sim_item_inner_ids]
        sim_item_ids = []
        for inner_id,weight in sim_item_inner_ids:
            sim_item_id = self.inner_id_to_item_ids[inner_id]
            #因为这个推荐系统会把自己也推荐出来
            if sim_item_id == item_id:
                continue
            sim_item_ids.append((sim_item_id,weight))
        return sim_item_ids

'''
model = ImplicitRecommend()

#items,matrix = read_dataframe(sys.argv[1])
df = pd.read_csv("~/Downloads/sw_order_product.csv",names=['user','item','weight'])
model.fit(df)
items = df.groupby('item')
output_filename = os.path.splitext(sys.argv[0])[0]+".csv"
with open(output_filename,"w") as f: 
    for item,_ in items:
        for sim_item_id,weight in model.get_similar_items(item):
            f.write("%s,%s,%s\n" % (item,sim_item_id,weight))
'''