from clickhouse_driver import Client
from impala.dbapi import connect
r = Client(host='172.24.115.18', port='8123', user='sunny', password='sunny')
conn = connect(host='172.24.115.19', port=10000, database='default',auth_mechanism='PLAIN', user='root',
            password='123456')
cur = conn.cursor()
cur.execute('use zcwzdss_db')
cur.execute('show tables')
cur.execute('select * from ods_zzll_bd')
print(r)
print(cur.fetchall())