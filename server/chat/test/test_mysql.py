
import mysql.connector

# 创建到 MySQL 服务器的连接
# cnx = mysql.connector.connect(user='chatbot_r', password='kW0LC5CnhKkA-',
#                               host='10.1.48.9', port=3306,
#                               database='chatbot')

cnx = mysql.connector.connect(user='chatbot_r', password='kW0LC5CnhKkA-',
                              host='sh-cynosdbmysql-grp-7oql582g.sql.tencentcdb.com', port=22457,
                              database='chatbot')

# 创建一个 cursor 对象
cursor = cnx.cursor()

# 执行一个查询
query = ("show tables")
# query = ("select * from feishu_evaluation limit 10")
cursor.execute(query)

# 从数据库获取数据
for row in cursor:
  print(row)

# 关闭 cursor 和连接
cursor.close()
cnx.close()