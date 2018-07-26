import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.data import Dataset
import pandas as pd
from annoy import AnnoyIndex

class LightFMRecommend:

    def __init__(self, *args, **kwargs):
        self.model = None
        self.annoy_idx = None
        self.inner_id_to_item_ids = None
        self.item_id_to_inner_ids = None
        
    
    def fit(self,df_source:pd.DataFrame):
        model = LightFM(loss='warp')

        dataset = Dataset()
        print("fit dataset")
        dataset.fit((x['user'] for _,x in df_source.iterrows()),(x['item'] for _,x in df_source.iterrows()))
        interactions,_ = dataset.build_interactions(((x['user'],x['item']) for _,x in df_source.iterrows()))
        print("fit model")
        self.item_id_to_inner_ids = dataset.mapping()[2]
        self.inner_id_to_item_ids = { v:k  for k,v in self.item_id_to_inner_ids.items()}
        model.fit(interactions,epochs=30,num_threads=4)
        _, item_embeddings = model.get_item_representations()
        print("insert into annoy")
        factors = item_embeddings.shape[1]
        self.annoy_idx = AnnoyIndex(factors)
        for i in range(item_embeddings.shape[0]):
            v = item_embeddings[i]
            self.annoy_idx.add_item(i,v)

        self.annoy_idx.build(10)
        self.annoy_idx.save('.ann')

    def get_similar_items(self,item_id,count: int=10):
        inner_id = self.item_id_to_inner_ids[item_id]
        inner_items_ids = self.annoy_idx.get_nns_by_item(inner_id,count)
        return [(self.inner_id_to_item_ids[inner],0) for inner in inner_items_ids]


# data = fetch_movielens(min_rating=5.0)

# train = data['train']
# test = data['test']


#print(precision_at_k(model,data['test'],k=5).mean())

# ranks = model.predict_rank(data['test'])
# print(ranks)

# def nearest_movies_annoy(movie_id,index,n=10,print_output=True):
#     nn = index.get_nns_by_item(movie_id,10)
#     if print_output:
#         print('Closest to %s : \n' % data['item_labels'][movie_id])
#     titles = [data['item_labels'][i] for i in nn]
#     if print_output:
#         print("\n".join(titles))
    
# nearest_movies_annoy(90,annoy_idx)
