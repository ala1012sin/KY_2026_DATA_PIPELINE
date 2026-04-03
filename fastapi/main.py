from datetime import datetime, timedelta
import logging
from pathlib import Path
import warnings
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from scheduler.sensor.scheduler import SensorScheduler
from scheduler.EW.scheduler import EWScheduler
from setting.database_orm import db_connection_pool, engine
from db.base import Base
from db.public.models import TB_CUSTOMER, TB_DEVICE, TB_VIBRATION_LOG, TB_FLOW_LOG, TB_WARN_ERROR_LOG, TB_PEMSPROPLUS_LOG, TB_PEMS_PRO_LOG, TB_AI_PEMS_LOG
from api.routers.predict_router import router as predict_router
from api.routers.simulate_router import router as simulate_router
from api.routers.model_router import router as model_router
from api.routers.monitoring_router import router as monitoring_router
from api.routers.optimize_router import router as optimize_router

load_dotenv()

# FastAPI 시작/종료 시점에 실행할 초기화/정리 로직
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ORM 메타데이터 기준으로 필요한 테이블 자동 생성
    Base.metadata.create_all(bind=engine)
    print("DB 테이블 생성 완료")
    
    # 앱 시작 시 스케줄러 실행
    scheduler.start()
    try:
        yield
    finally:
        # 앱 종료 시 스케줄러 중지
        scheduler.shutdown(wait=False)


app = FastAPI(lifespan=lifespan)
scheduler = AsyncIOScheduler()

# 정적 웹 파일(simulate.html 등) 경로 준비 및 마운트
WEB_DIR = Path(__file__).resolve().parent / "web"
WEB_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")

# 예측 라우터 추가
app.include_router(predict_router, prefix="/api", tags=["predict"])
app.include_router(simulate_router, prefix="/api", tags=["simulate"])
app.include_router(model_router, prefix="/api", tags=["model"])
app.include_router(monitoring_router, prefix="/api", tags=["monitoring"])
app.include_router(optimize_router, prefix="/api", tags=["optimize"])


@app.get("/", include_in_schema=False)
def index():
    # 루트 접속 시 시뮬레이션 페이지 우선 제공
    html_path = WEB_DIR / "simulate.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    return RedirectResponse(url="/docs")


@app.get("/simulate", include_in_schema=False)
def simulate_page():
    # 시뮬레이션 페이지가 없으면 Swagger 문서로 리다이렉트
    html_path = WEB_DIR / "simulate.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    return RedirectResponse(url="/docs")


@app.get("/milp", include_in_schema=False)
def milp_page():
    # MILP 피크 분배 페이지가 없으면 Swagger 문서로 리다이렉트
    html_path = WEB_DIR / "milp.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    return RedirectResponse(url="/docs")


async def _run_sensor_job(minutes: int = 1) -> None:
    # 최근 N분 구간을 기준으로 센서 배치 실행
    tz = ZoneInfo("Asia/Seoul")
    end = datetime.now(tz=tz)
    start = end - timedelta(minutes=minutes)

    # 제너레이터 기반 DB 세션 획득/정리
    db_gen = db_connection_pool()
    db = next(db_gen)
    try:
        SensorScheduler(db).run(start, end)
    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass
        
# 아직 에러 데이터 수신이 되지 않아 주석 처리
#async def _run_ew_job(minutes: int = 1) -> None:
#    # 최근 N분 구간을 기준으로 EW 배치 실행
#    tz = ZoneInfo("Asia/Seoul")
#    end = datetime.now(tz=tz)
#    start = end - timedelta(minutes=minutes)
#
#    db_gen = db_connection_pool()
#    db = next(db_gen)
#    try:
#        EWScheduler(db).run(start, end)
#    finally:
#        try:
#            next(db_gen)
#        except StopIteration:
#            pass



@scheduler.scheduled_job("interval", seconds=60)
async def sensor_cron_job():
    # 60초마다 최근 1분 데이터 처리
    await _run_sensor_job(minutes=1)

# 아직 에러 데이터 수신이 되지 않아 주석 처리
#@scheduler.scheduled_job("interval", minutes=1) 
#async def ew_cron_job():
#    # 1분마다 최근 1분 EW 데이터 처리
#    try:
#        await _run_ew_job(minutes=1)
#    except Exception as e:
#        logging.exception(f"EW 스케줄러 실행 실패: {e}")
