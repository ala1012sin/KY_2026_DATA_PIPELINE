from datetime import datetime, timedelta
import logging
import warnings
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from contextlib import asynccontextmanager

from scheduler.sensor.scheduler import SensorScheduler
from setting.database_orm import db_connection_pool, engine
from db.base import Base
from db.public.models import TB_CUSTOMER, TB_DEVICE, TB_VIBRATION_LOG, TB_FLOW_LOG, TB_WARN_ERROR_LOG, TB_PEMSPROPLUS_LOG, TB_PEMS_PRO_LOG, TB_AI_PEMS_LOG
from api.routers.predict_router import router as predict_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    print("DB 테이블 생성 완료")
    
    scheduler.start()
    try:
        yield
    finally:
        scheduler.shutdown(wait=False)


app = FastAPI(lifespan=lifespan)
scheduler = AsyncIOScheduler()

# 예측 라우터 추가
app.include_router(predict_router, prefix="/api", tags=["predict"])


async def _run_sensor_job(minutes: int = 1) -> None:
    tz = ZoneInfo("Asia/Seoul")
    end = datetime.now(tz=tz)
    start = end - timedelta(minutes=minutes)

    db_gen = db_connection_pool()
    db = next(db_gen)
    try:
        SensorScheduler(db).run(start, end)
    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass


@scheduler.scheduled_job("interval", seconds=60)
async def sensor_cron_job():
    await _run_sensor_job(minutes=1)

