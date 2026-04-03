from datetime import datetime
from typing import Optional, Union

import os
import time
import requests
from dotenv import load_dotenv

from Logger import Logger as logger
from db.public.models import TB_DEVICE, TB_WARN_ERROR_LOG
from infrastructure.queryFactory.base_orm import BaseQueryFactory
from service.ew_code_mapper import decode_ew_code_text

load_dotenv()

class EWScheduler:
    """Error/Warning 데이터 API 스케줄러 클래스"""
    _ZERO_CODES = {"0x0000000000000000", "0x0", "0"}

    def __init__(self, db_session):
        """
        스케줄러 초기화

        Args:
            db_session: SQLAlchemy DB Session
        """
        self.db = db_session
        self.logger = logger
        self.base_url = os.getenv("KY_ERROR_DATA_URL", "").strip()
        self.api_token = os.getenv("KY_API_TOKEN", "").strip()
        self.timeout_sec = float(os.getenv("KY_API_TIMEOUT", "10"))
        self.retries = int(os.getenv("KY_API_RETRIES", "2"))
        self.retry_backoff_sec = float(os.getenv("KY_API_RETRY_BACKOFF", "1.0"))
        self.http = requests.Session()
        self.http.headers.update({"Accept": "application/json"})
        if self.api_token:
            self.http.headers.update({"Authorization": f"Bearer {self.api_token}"})

    def _safe_rollback(self) -> None:
        """DB 예외 후 세션을 정상 상태로 되돌린다."""
        try:
            self.db.rollback()
        except Exception as rollback_error:
            self.logger.error(f"EW 스케줄러 롤백 실패: {rollback_error}")

    def _parse_datetime(self, dt_str: any) -> Optional[datetime]:
        """API 응답 시간 문자열을 datetime으로 변환"""
        if not dt_str:
            return None
        if isinstance(dt_str, str):
            return datetime.strptime(dt_str, "%Y%m%d%H%M%S")
        return dt_str

    def _find_device_by_identity(self, serial_no: str, device_type: int, device_num: Union[int, str]):
        """serial_no + device_type + device_num 조합으로 디바이스 조회"""
        try:
            return (
                self.db.query(TB_DEVICE)
                .filter(
                    TB_DEVICE.serial_no == serial_no,
                    TB_DEVICE.device_type == str(device_type),
                    TB_DEVICE.device_num == str(device_num),
                )
                .first()
            )
        except Exception:
            self._safe_rollback()
            raise

    def _normalize_code(self, code: any) -> Optional[str]:
        """에러/경보 코드 정규화 (유효하지 않으면 None)"""
        if code is None:
            return None
        code_str = str(code).strip()
        if not code_str or code_str.lower() in self._ZERO_CODES:
            return None
        return code_str

    def _find_existing_log(
        self, device_id, ew_dt: datetime, code: str, error_warn: int
    ) -> Optional[TB_WARN_ERROR_LOG]:
        try:
            return (
                self.db.query(TB_WARN_ERROR_LOG)
                .filter(
                    TB_WARN_ERROR_LOG.device_id == device_id,
                    TB_WARN_ERROR_LOG.ew_dt == ew_dt,
                    TB_WARN_ERROR_LOG.code == code,
                    TB_WARN_ERROR_LOG.error_warn == error_warn,
                )
                .first()
            )
        except Exception:
            self._safe_rollback()
            raise

    def _insert_ew_log(
        self,
        device_id,
        ew_dt: datetime,
        code: str,
        error_warn: int,
        ew_note: Optional[str] = None,
    ) -> bool:
        existing = self._find_existing_log(device_id, ew_dt, code, error_warn)
        if existing:
            return False

        log_query = BaseQueryFactory(self.db, TB_WARN_ERROR_LOG)
        log_query.insert_single_row(
            device_id=device_id,
            ew_dt=ew_dt,
            code=code,
            error_warn=error_warn,
            ew_note=ew_note,
        )
        return True

    def fetch_api_data(self, start_dt: datetime, end_dt: datetime, limit: int = 50000) -> list:
        """
        API에서 Error/Warning 데이터 조회

        Args:
            start_dt: 시작 시간 (datetime)
            end_dt: 종료 시간 (datetime)
            limit: 조회 데이터 한계

        Returns:
            API 응답 데이터 리스트
        """
        params = {
            "startDate": start_dt.strftime("%Y%m%d%H%M%S"),
            "endDate": end_dt.strftime("%Y%m%d%H%M%S"),
            "limit": limit,
        }

        if not self.base_url:
            self.logger.error("EW API 수신 실패: KY_ERROR_DATA_URL 환경변수가 비어 있습니다")
            return []

        last_err = None

        for attempt in range(self.retries + 1):
            try:
                response = self.http.get(self.base_url, params=params, timeout=self.timeout_sec)
                self.logger.info(f"EW API 요청: {response.url}")

                if response.status_code == 200:
                    if not response.text or not response.text.strip():
                        self.logger.info("EW API 응답: 200 (데이터 없음)")
                        return []
                    try:
                        data = response.json()
                    except ValueError as e:
                        preview = response.text[:300].replace("\n", " ")
                        self.logger.error(f"EW API JSON 파싱 실패: {e} / body={preview}")
                        return []

                    if not isinstance(data, list):
                        self.logger.error(f"EW API 응답 형식 오류: list가 아니라 {type(data).__name__}")
                        return []

                    self.logger.info(f"EW API 응답: 200 (데이터 {len(data)}건)")
                    return data

                if response.status_code == 404:
                    self.logger.error("EW API 응답: 404 (엔드포인트 없음) → Base URL/경로 변경 의심")
                    return []

                if response.status_code in (401, 403):
                    self.logger.error(f"EW API 응답: {response.status_code} (인증/권한 문제)")
                    return []

                preview = response.text[:300].replace("\n", " ")
                self.logger.error(f"EW API 응답: HTTP {response.status_code} (요청 실패) / body={preview}")
                return []
            except requests.RequestException as e:
                last_err = e
                if attempt < self.retries:
                    wait = self.retry_backoff_sec * (attempt + 1)
                    self.logger.warning(
                        f"EW API 통신 오류: {type(e).__name__} → {wait:.1f}초 후 재시도 ({attempt+1}/{self.retries})"
                    )
                    time.sleep(wait)
                    continue

                self.logger.error(f"EW API 최종 실패 (재시도 종료): {type(e).__name__}: {e}")
                return []
            except Exception as e:
                self.logger.error(f"EW API 수신 처리 중 예외 발생: {type(e).__name__}: {e}")
                return []

        self.logger.error(f"EW API 최종 실패: {last_err}")
        return []

    def process_ew_data(self, data: list) -> int:
        """
        EW 데이터 처리 및 적재

        Args:
            data: EW 데이터 리스트

        Returns:
            성공적으로 적재된 데이터 개수
        """
        success_count = 0

        for item in data:
            try:
                device_info = item.get("deviceInfo", {})
                serial_no = device_info.get("serialNo")
                device_num = device_info.get("deviceNum")
                device_type = device_info.get("deviceType")

                if not serial_no:
                    self.logger.warning("EW 데이터에 serialNo가 없습니다. 스킵")
                    continue

                if device_type is None or device_num is None:
                    self.logger.warning(
                        f"EW 데이터에 deviceType/deviceNum이 없습니다. serial_no={serial_no}"
                    )
                    continue

                device = self._find_device_by_identity(serial_no, device_type, device_num)
                if not device:
                    self.logger.warning(
                        f"디바이스 미존재: serial_no={serial_no}, device_type={device_type}, device_num={device_num}"
                    )
                    continue

                ew_dt = self._parse_datetime(item.get("dt"))
                if not ew_dt:
                    self.logger.warning(
                        f"EW 데이터에 dt가 없습니다. serial_no={serial_no}, device_num={device_num}"
                    )
                    continue

                error_code = self._normalize_code(item.get("error"))
                warn_code = self._normalize_code(item.get("warn"))
                error_note_raw = item.get("errorNote") or item.get("error_note")
                warn_note_raw = item.get("warnNote") or item.get("warn_note")

                error_note = (str(error_note_raw).strip() if error_note_raw is not None else "")
                warn_note = (str(warn_note_raw).strip() if warn_note_raw is not None else "")

                if error_code and not error_note:
                    error_note = decode_ew_code_text(error_code)
                if warn_code and not warn_note:
                    warn_note = decode_ew_code_text(warn_code)

                if error_code:
                    inserted = self._insert_ew_log(
                        device_id=device.device_id,
                        ew_dt=ew_dt,
                        code=error_code,
                        error_warn=0,
                        ew_note=(error_note or None),
                    )
                    if inserted:
                        success_count += 1

                if warn_code:
                    inserted = self._insert_ew_log(
                        device_id=device.device_id,
                        ew_dt=ew_dt,
                        code=warn_code,
                        error_warn=1,
                        ew_note=(warn_note or None),
                    )
                    if inserted:
                        success_count += 1

            except Exception as e:
                self._safe_rollback()
                self.logger.error(f"EW 데이터 적재 중 에러 발생: {e}")
                continue

        return success_count

    def run(self, start_dt: datetime, end_dt: datetime):
        """
        스케줄러 실행

        Args:
            start_dt: 시작 시간 (datetime)
            end_dt: 종료 시간 (datetime)
        """
        self.logger.info("EW 스케줄러 시작")

        data = self.fetch_api_data(start_dt, end_dt)
        if not data:
            self.logger.warning("처리할 EW 데이터가 없습니다.")
            return

        success = self.process_ew_data(data)
        self.logger.info(f"EW 데이터 처리 완료: {success}/{len(data)}")
        self.logger.info("EW 스케줄러 실행 완료")


## 고장,경보에대한 정보는 AI PEMS, PEMS Pro에 대한 것인데 pems pro Plus에 대한 정보를 따로 있는지 확인 되면 더 하기
