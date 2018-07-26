import pymysql.cursors
import pandas as pd

def _get_db_connection(conf):
    connection = pymysql.connect(host=conf['db_host'],
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
    with connection.cursor() as cursor:
        for item_id,sim in data.items():
            if conf['mode'] == 'row':
                for sim_item,weight in sim:
                    cursor.execute(conf['insert_sql'] % (item_id,sim_item))

            elif conf['mode'] == 'dot':
                pass
            elif conf['mode'] == 'json':
                pass
    connection.commit()

def read_from_csv(path):
    return pd.read_csv(path,names=['user','item','weight'])