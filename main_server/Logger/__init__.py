import logging

# Logging 설정
logging.basicConfig(
    level=logging.INFO,  # 로그 레벨 설정 (DEBUG, INFO, WARNING, ERROR, CRITICAL 중 선택)
    format='%(asctime)s - %(levelname)s - %(message)s',  # 로그 메시지 형식
    datefmt='%Y-%m-%d %H:%M:%S',  # 날짜 형식
    handlers=[
        logging.FileHandler("application.log"),  # 로그를 기록할 파일 이름
        logging.StreamHandler()  # 콘솔 출력도 가능하게 설정
    ]
)

# root logger를 반환
Logger = logging.getLogger()