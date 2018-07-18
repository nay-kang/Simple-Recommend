from surprise import SVD , KNNBaseline , NMF , SlopeOne , NormalPredictor , KNNBasic , CoClustering , BaselineOnly
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
import pandas as pd
import math
import os
import sys

class SurpriseRecommend:

    def __init__(self):
        self.algo = None

    def fit(self,df_source: pd.DataFrame):
        '''
        df_source: 数据类型是pandas DataFrame，里面的数据结构是 user/item/weight
        '''
        reader = Reader(rating_scale=(0,1))
        data = Dataset.load_from_df(df_source,reader=reader)
        sim_options = {'name':'cosine','user_based':False}
        self.algo = KNNBaseline(sim_options=sim_options)
        trainset = data.build_full_trainset()
        self.algo.fit(trainset)

    def get_similar_items(self,item_id, count: int=10):
        inner_item_id = self.algo.trainset.to_inner_iid(item_id)
        sim_item_inner_ids = self.algo.get_neighbors(inner_item_id,count)
        return (self.algo.trainset.to_raw_iid(inner_id) for inner_id in sim_item_inner_ids)

model = SurpriseRecommend()

file_path = os.path.expanduser(sys.argv[1])
# file_path = "~/Downloads/sw_order_product.csv"
df = pd.read_csv(file_path,names=["user","item","weight"])
model.fit(df)

items = df.groupby('item')
output_filename = os.path.splitext(sys.argv[0])[0]+".csv"
with open(output_filename,"w") as f: 
    for item,_ in items:
        f.write("%s,%s,%s\n" % (item,item,'NaN'))
        for sim_item_id in model.get_similar_items(item):
            f.write("%s,%s,%s\n" % (item,sim_item_id,'NaN'))