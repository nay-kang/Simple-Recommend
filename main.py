from implicit_github import ImplicitRecommend
from surprise_github import SurpriseRecommend
from random_rec import RandomRecommend
from lightfm_github import LightFMRecommend
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math
from multiprocessing import Pool
from yaml import load
from data_source import read_from_database,write_to_database,read_from_csv
from datetime import datetime

MODELS = {
    "implicit":ImplicitRecommend,
    "surprise":SurpriseRecommend,
    "random":RandomRecommend,
    'lightfm':LightFMRecommend,
}

def get_similar(model,df,output_filename,count=10):
    '''
    生成item-item的推荐csv文件
    并返回item的数组词典
    '''
    result = {}
    items = df.groupby('item')
    with open(output_filename,"w") as f: 
        for item,_ in items:
            #f.write("%s,%s,%s\n" % (item,item,1))
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
    print('split train set')
    train_df,test_df = train_test_split(full_df,test_size=0.2,stratify=full_df['user'])
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    #并存入临时文件，方便debug
    train_df.to_csv('tmp_train_set.csv')
    test_df.to_csv('tmp_test_set.csv')
    
    #训练模型，并得到推荐商品
    print('get similar data')
    model.fit(train_df)
    full_sim_items = get_similar(model,train_df,'tmp_similar.csv',count*2)
    train_df_idx = train_df.set_index('user')
    test_users = test_df.groupby('user')
    
    # 计算数据
    print('calc metric')
    pass_user = 0
    metric = {}
    total_hit = 0
    item_in_test_count = 0
    '''
    multi process
    '''
    fn_params = []
    print('gather params')
    for user,group in test_users:
        fn_params.append((train_df_idx,full_sim_items,user,group,count))

    print('run multi process')
    with Pool(4) as pool:
        rtns = pool.map(_evaluate,fn_params)
    print('gather return')
    for user,m,recall,pass_u in rtns:
        total_hit += m['hit']
        pass_user += pass_u
        if pass_u == 0:
            metric[user] = m
        item_in_test_count += recall
    

    '''
    single process
    
    for user,group in test_users:
        _,m,recall,pass_u = _evaluate((train_df_idx,full_sim_items,user,group,count))
        total_hit += m['htt']
        pass_user += pass_u
        if pass_u == 0:
            metric[user] = m
        item_in_test_count += recall
    '''

    return (total_hit/(len(metric)*count)),total_hit/item_in_test_count,metric,pass_user

def _evaluate(v):
    train_df_idx = v[0]
    full_sim_items = v[1]
    user = v[2]
    group = v[3]
    count = v[4]
    pass_user = 0
    try:
        item_in_train = list(train_df_idx.loc[user]['item'])
    except:
        #因为训练集中无和用户相关的商品，所以跳过
        return user,None,0,1
    
    item_in_test = list(group['item'])
    #因为测试集中无和用户关联的商品，所以跳过
    if len(item_in_test) <=0:
        return user,None,0,1

    sim_items = _get_sim_item_by_items_2(item_in_train,full_sim_items)
    # 排重
    dup_idx = np.unique(sim_items,return_index=True)[1]
    sim_items = [sim_items[index] for index in sorted(dup_idx)]
    # recall_count = len(sim_items)
    # 移除在训练集里面重复的
    for item in item_in_train:
        try:
            sim_items.remove(item)
        except:
            pass
    # 截断出topN
    sim_items = sim_items[:count]
    #获得命中次数
    hit = 0
    for iit in item_in_test:
        hit += int(iit in sim_items)
    #metric[user] = {"hit":hit,"hit_percent":hit/count,"recall":hit/len(item_in_test)}
    return user,{"hit":hit,"hit_percent":hit/count,"recall":hit/len(item_in_test)},len(item_in_test),pass_user
    # total_hit += hit
    # item_in_test_count += len(item_in_test)

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

def _get_sim_item_by_items_2(origin_items,full_similar_items):
    '''
    思路：
        origin_items => s_a,s_b,s_c
        三个商品的相关商品
        [s_a,[a1,a2,a3,a4,a5]]
        [s_b,[a1,a3,a2,a5,a7]]
        [s_c,[a3,a2,a7,a9,a1]]
        推导出
        [a1,a3,a2,a7,a5,...]
        属于互相叠加的
    '''    
    sim_items_with_value = {}
    for origin_item in origin_items:
        try:
            sim_items = full_similar_items[origin_item]
        except:
            continue
        for idx in range(len(sim_items)):
            sim_item = sim_items[idx][0]
            value = 1/(math.sqrt(idx+1))
            if sim_item in sim_items_with_value:
                sim_items_with_value[sim_item] += value
            else:
                sim_items_with_value[sim_item] = value

    rtn_sim_items = []
    for key,_val in sorted(sim_items_with_value.items(), key=lambda kv: kv[1],reverse=True):
        rtn_sim_items.append(key)
    
    return rtn_sim_items

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple Recommend")
    parser.add_argument('command',help='gen_similar,evaluate')
    parser.add_argument('model',help='available models are %s' % ",".join(MODELS.keys()))
    parser.add_argument('--read_from_csv',help='csv file to fit')
    parser.add_argument('--read_from_db',help='db_source.yml config path')
    parser.add_argument('--write_to_db',help='db_target.yml config path')
    parser.add_argument('--evaluate_times',default=5,help='how many times evaluate repeat')
    parser.add_argument('--item_count',default=50,help='how many items return while evaluate')

    args = parser.parse_args()
    model_class = MODELS[args.model]
    model = model_class()

    if args.read_from_csv:
        csv_file = args.read_from_csv
        df = read_from_csv(csv_file)
    
    if args.read_from_db:
        with open(args.read_from_db) as f:
            conf = load(f.read())
        df = read_from_database(conf)
    
    # 筛选交互超过20个用户的商品
    count_df = df.groupby('item').size()
    enough_df = count_df[count_df >= 20].reset_index()[['item']]
    df = df.merge(enough_df,how='right',left_on='item',right_on='item')
    #筛选大于5个商品的用户
    count_df = df.groupby(['user','item']).size().groupby('user').size()
    enough_df = count_df[count_df >= 5].reset_index()[['user']]
    df = df.merge(enough_df,how='right',left_on='user',right_on='user')
    
    if args.command == 'gen_similar':
        model.fit(df)
        output_filename = 'output_%s_%s.csv' % (args.model,datetime.now().strftime('%Y%m%d_%H%M%S'))
        similars = get_similar(model,df,output_filename,args.item_count)
        if args.write_to_db:
            with open(args.write_to_db) as f:
                target_conf = load(f.read())
            write_to_database(target_conf,similars)
    
    if args.command == 'evaluate':
        result = []
        for i in range(int(args.evaluate_times)):
            precision,recall,metric,passed = evaluate(model,df,int(args.item_count))
            result.append((precision,recall,metric,passed))
        for r in result:
            print("user:%s \t passed:%s \tprecision:%2.4f%% \trecall:%2.4f%%" % (len(r[2]),r[3],r[0]*100,r[1]*100) )