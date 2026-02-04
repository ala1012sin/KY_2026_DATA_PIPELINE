# infrastructure/queryFactory/base_orm.py
"""
ORM 기반 DB QueryFactory 모듈
- 해당 모듈 활용 DB 테이블별 CRUD 수행
"""
from error.errors import DataBaseError
from sqlalchemy.orm import Session
from typing import Type, TypeVar, Generic, Optional, List,Tuple, Any
from sqlalchemy import and_
from Logger import Logger as logger
# TypeVar를 사용하여 모델 타입을 제네릭으로 지정
T = TypeVar("T")

class BaseQueryFactory(Generic[T]):
    def __init__(self,conn:Session, model: Type[T]):
        self.conn = conn
        self.model = model
        
    def find_one(self, **filters) -> Optional[T]:
        try:
            return self.conn.query(self.model).filter_by(**filters).one()
        except:
            self.conn.rollback()
            return None
    def find_all(self, **filters) -> List[T]:
        try:
            return self.conn.query(self.model).filter_by(**filters).all()
        except:
            self.conn.rollback()
            return None
    def insert_single_row(self,**data) -> T:
        try:
            instance = self.model(**data)
            self.conn.add(instance)
            self.conn.commit()
            self.conn.refresh(instance)
            return instance
        except Exception as e:
            self.conn.rollback()
            logger.error(f"DB Error {e}")
            raise DataBaseError(message="데이터베이스 에러 발생")
    
    def insert_multi_row(self,data_lst:List[T]) -> T:
        try:
            self.conn.add_all(data_lst)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            logger.error(f"DB Error {e}")
            raise DataBaseError(message="데이터베이스 에러 발생")
    
    def update(self, instance: T, **data) -> T:
        try:
            for key, value in data.items():
                setattr(instance, key, value)
            self.conn.commit()
            self.conn.refresh(instance)
            return instance
        except Exception as e:
            self.conn.rollback()
            logger.error(f"DB Error {e}")
            raise DataBaseError(message="데이터베이스 에러 발생")
        
    def _find_device_by_identity(self, serial_no: str, device_type: int, device_num: int | str):
        """serial_no + device_type + device_num 조합으로 디바이스 조회"""
        return (
            self.conn.query(self.model)
            .filter(
                self.model.serial_no == serial_no,
                self.model.device_type == str(device_type),
                self.model.device_num == str(device_num),
            )
            .first()
        )
    
