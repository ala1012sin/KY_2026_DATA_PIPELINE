# 1. 공식 PostgreSQL 15버전 이미지를 가져옵니다.
FROM postgres:latest

# 2. DB 시간도 한국 시간으로 맞춥니다. (데이터 저장 시간 중요하니까요)
ENV TZ=Asia/Seoul

# 3. 우리가 만든 SQL 파일을 특수 폴더에 복사합니다.
# /docker-entrypoint-initdb.d/ 안에 sql 파일을 넣어두면
# 컨테이너가 처음 실행될 때 알파벳 순서대로 자동 실행됩니다.
COPY init.sql /docker-entrypoint-initdb.d/