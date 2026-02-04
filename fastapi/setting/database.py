# """동기 방식 DB  연결 정의"""
# import pg8000
# from Logger import Logger as logger
# import os

# def db_connection_pool():
#     logger.info(">>>> DataBase Setting Start  <<<<")
#     logger.info('postgre connection Start')
#     conn = pg8000.connect(
#         host=os.getenv("DB_HOST"),
#         user=os.getenv("DB_USER"),
#         password=os.getenv("DB_PASSWORD"),
#         port=os.getenv("DB_PORT"),
#         database=os.getenv("DB_NAME")
#     )
#     return conn