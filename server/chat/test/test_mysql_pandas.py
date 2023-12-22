import pandas as pd
from sqlalchemy import create_engine

# 创建数据库连接
engine = create_engine('mysql+mysqlconnector://chatbot_r:kW0LC5CnhKkA-@10.1.48.9:3306/chatbot')

# # 使用pandas的read_sql_query函数直接读取mysql中的数据
df = pd.read_sql_query('SELECT * FROM feishu_evaluation', engine)

print(df)



# 使用pandas的read_sql_query函数直接读取mysql中的数据
df_all = pd.read_sql_query('SELECT * from feishu_message', engine)

print(df_all)

merge_df = pd.merge(df_all, df, left_on="card_id", right_on="msg_id", how='left')

merge_df.to_excel("evaluate.xlsx")