import numpy as np
import pandas as pd

class RandomRecommend:

    def __init__(self):
        self.items = None

    def fit(self,df_source: pd.DataFrame):
        '''
        df_source: 数据类型是pandas DataFrame，里面的数据结构是 user/item/weight
        '''
        self.items = list(df_source.groupby('item').groups.keys())
        print("fit random model")

    def get_similar_items(self,item_id, count: int=10):
        np.random.shuffle(self.items)
        l = self.items[:count+1]
        try:
            l.delete(item_id)
        except:
            pass
        l = l[:count]
        return [(sim_item_id,0) for sim_item_id in l]

