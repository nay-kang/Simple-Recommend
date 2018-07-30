# import pymysql.cursors
from mysql import connector
import pandas as pd
from datetime import datetime
import json

def _get_db_connection(conf):
    connection = connector.connect(host=conf['db_host'],
                                user=conf['db_user'],
                                password=conf['db_password'],
                                db=conf['db_database'],
                                port=conf['db_port'])
    return connection

def read_from_database(conf):
    connection = _get_db_connection(conf)  
    df = pd.read_sql(conf['fetch_sql'],connection)
    return df

def write_to_database(conf,data):
    connection = _get_db_connection(conf)
    cursor = connection.cursor()
    if 'before_sql' in conf:
        cursor.execute(conf['before_sql'])
    for item_id,sim in data.items():
        if conf['mode'] == 'row':
            for sim_item,weight in sim:
                cursor.execute(conf['insert_sql'] % (item_id,sim_item))

        elif conf['mode'] == 'dot':
            pass
        elif conf['mode'] == 'json':
            cursor.execute(conf['insert_sql'] % (item_id,json.dumps([sim_item for sim_item,_ in sim])))
            pass
    connection.commit()

def read_from_csv(path):
    return pd.read_csv(path,names=['user','item','weight'])