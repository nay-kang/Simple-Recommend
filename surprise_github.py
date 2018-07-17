from surprise import SVD , KNNBaseline , NMF , SlopeOne , NormalPredictor , KNNBasic , CoClustering , BaselineOnly
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
import pandas as pd
import math
import os
import sys

file_path = os.path.expanduser(sys.argv[1])
# file_path = "~/Downloads/sw_order_product.csv"
df = pd.read_csv(file_path,names=["user","item","weight"])

reader = Reader(rating_scale=(0,1))
data = Dataset.load_from_df(df,reader=reader)

sim_options = {'name':'cosine','user_based':False}
algo = KNNBaseline(sim_options=sim_options)

#cross_validate(algo,data,measures=['RMSE','MAE'],cv=5,verbose=True)

trainset = data.build_full_trainset()
algo.fit(trainset)

items = df.groupby("item")

output_filename = os.path.splitext(sys.argv[0])[0]+".csv"
with open(output_filename,"w") as f: 
    for item,_group in items:
        item_iid = algo.trainset.to_inner_iid(item)
        neighbor_iids = algo.get_neighbors(item_iid,k=10)
        neighbor_item_ids = (algo.trainset.to_raw_iid(inner_id) for inner_id in neighbor_iids)
        f.write("%s,%s,%s\n" % (item,item,'NaN'))
        for inner_id in neighbor_iids:
            raw_id = algo.trainset.to_raw_iid(inner_id)
            f.write("%s,%s,%s\n" % (item,raw_id,'NaN'))
        # for raw_id in neighbor_item_ids:
        #     f.write("%s,%s,%s\n" % (item,raw_id,'NaN'))

