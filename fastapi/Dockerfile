# 1. 파이썬 3.10 슬림 버전(가벼운 버전)을 기반으로 이미지를 만듭니다.
FROM python:3.10.13

# 2. 컨테이너 내부의 작업 폴더를 /app으로 정합니다.
WORKDIR /app

# 3. 리눅스 시스템 시간 설정을 위한 패키지 설치 (이게 없으면 시간이 9시간 차이남)
# apt-get은 리눅스 설치 명령어입니다.
RUN apt-get update && apt-get install -y tzdata postgresql-client \
    && rm -rf /var/lib/apt/lists/*
# 환경변수 TZ를 한국 시간으로 고정합니다.
ENV TZ=Asia/Seoul

# 4. requirements.txt를 먼저 복사해서 라이브러리를 설치합니다.
# (코드가 바뀔 때마다 라이브러리를 다시 까는 것을 방지하기 위해 캐싱)
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# 5. 현재 폴더의 모든 파일(main.py 등)을 컨테이너 내부 /app 폴더로 복사합니다.
COPY . .

# 6. 컨테이너가 켜지면 실행할 명령어입니다.
# 0.0.0.0: 외부(내 컴퓨터)에서 접속 가능하게 엶
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]