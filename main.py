from implicit_github import ImplicitRecommend
from surprise_github import SurpriseRecommend
from random_rec import RandomRecommend
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

MODELS = {
    "implicit":ImplicitRecommend,
    "surprise":SurpriseRecommend,
    "random":RandomRecommend,
}

def read_csv(path):
    return pd.read_csv(path,names=['user','item','weight'])

def gen_similar(model,df,output_filename,count=10):
    '''
    生成item-item的推荐csv文件
    并返回item的数组词典
    '''
    result = {}
    items = df.groupby('item')
    with open(output_filename,"w") as f: 
        for item,_ in items:
            f.write("%s,%s,%s\n" % (item,item,1))
            result[item] = []
            for sim_item_id,weight in model.get_similar_items(item,count):
                f.write("%s,%s,%s\n" % (item,sim_item_id,weight))
                result[item].append((sim_item_id,weight))
    return result

def evaluate(model,full_df,count):
    '''
    model:配置好，但是还没有训练的模型
    full_df:全量的数据
    count:生成推荐的数量
    '''

    #分割训练集和测试集
    train_df,test_df = train_test_split(full_df,test_size=0.2,stratify=full_df['user'])
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    #并存入临时文件，方便debug
    train_df.to_csv('tmp_train_set.csv')
    test_df.to_csv('tmp_test_set.csv')
    
    #训练模型，并得到推荐商品
    model.fit(train_df)
    full_sim_items = gen_similar(model,train_df,'tmp_similar.csv',count*2)
    train_df_idx = train_df.set_index('user')
    test_users = test_df.groupby('user')
    

    pass_user = 0
    metric = {}
    total_hit = 0
    for user,group in test_users:
        try:
            item_in_train = list(train_df_idx.loc[user]['item'])
        except:
            #因为训练集中无和用户相关的商品，所以跳过
            pass_user += 1
            continue
        
        item_in_test = list(group['item'])
        #因为测试集中无和用户关联的商品，所以跳过
        if len(item_in_test) <=0:
            pass_user += 1
            continue

        sim_items = _get_sim_item_by_items_1(item_in_train,full_sim_items)
        # 排重
        dup_idx = np.unique(sim_items,return_index=True)[1]
        sim_items = [sim_items[index] for index in sorted(dup_idx)]
        recall_count = len(sim_items)
        # 移除在训练集里面重复的
        for item in item_in_train:
            try:
                sim_items.remove(item)
            except:
                pass
        # 截断出topN
        sim_items = sim_items[:count]
        hit = 0
        for iit in item_in_test:
            hit += int(iit in sim_items)
        metric[user] = {"hit":hit,"hit_percent":hit/count,"recall":hit/len(item_in_test)}
        total_hit += hit
    return (total_hit/(len(metric)*count)),metric,pass_user


def _get_sim_item_by_items_1(origin_items,full_similar_items):
    '''
    思路：
        origin_items => s_a,s_b,s_c
        三个商品的相关商品
        [s_a,[a1,a2,a3,a4,a5]]
        [s_b,[b1,b2,b3,b4,b5]]
        [s_c,[c1,c2,c3,c4,c5]]
        推导出
        [a1,b1,c1,a2,b2,c2,a3,b3,c3,...]
    '''
    # 先做一次乱序，避免有的产品永远按照顺序排在前面
    np.random.shuffle(origin_items)

    pick_sim_items = []
    for origin_item in origin_items:
        try:
            pick_sim_items.append(full_similar_items[origin_item])
        except:
            pass

    rtn_sim_items = []
    
    idx = 0
    has_next = True
    while has_next:
        _has_next = False
        for pick_sim_item in pick_sim_items:
            try:
                sim_item = pick_sim_item[idx][0]
                _has_next = True
                rtn_sim_items.append(sim_item)
            except:
                continue

        has_next = _has_next
        idx += 1
    return rtn_sim_items


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple Recommend")
    parser.add_argument('command',help='gen_similar,evaluate')
    parser.add_argument('model',help='available models are %s' % ",".join(MODELS.keys()))
    parser.add_argument('file',help='csv file to fit')
    parser.add_argument('--evaluate_times',default=5,help='how many times evaluate repeat')
    parser.add_argument('--evaluate_item_count',default=50,help='how many items return while evaluate')

    args = parser.parse_args()
    model_class = MODELS[args.model]
    model = model_class()
    csv_file = args.file
    df = read_csv(csv_file)
    #筛选大于5个商品的用户
    count_df = df.groupby(['user','item']).size().groupby('user').size()
    enough_df = count_df[count_df >= 5].reset_index()[['user']]
    df = df.merge(enough_df,how='right',left_on='user',right_on='user')
    if args.command == 'gen_similar':
        model.fit(df)
        output_filename = 'output_%s.csv' % args.model
        gen_similar(model,df,output_filename)
    
    if args.command == 'evaluate':
        result = []
        for i in range(int(args.evaluate_times)):
            percent,metric,passed = evaluate(model,df,int(args.evaluate_item_count))
            result.append((percent,metric,passed))
        for r in result:
            print("user:%s \t passed:%s \t%2.4f%%" % (len(r[1]),r[2],r[0]) )