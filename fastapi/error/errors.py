"""
사용자 직접 정의 Custom Error 정의
- FastAPI APP 최종 Endpoint 유저가 알아야할 에러가 있는 경우 해당 모듈에서 정의
"""
# DB Error
class DataBaseError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"DataBaseError: {self.message}"

    def __reduce__(self):
        return (self.__class__, (self.message,))