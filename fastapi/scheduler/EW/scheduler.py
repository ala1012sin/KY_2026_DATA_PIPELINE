from datetime import datetime

import requests

from Logger import Logger as logger
from db.public.models import TB_DEVICE, TB_WARN_ERROR_LOG
from infrastructure.queryFactory.base_orm import BaseQueryFactory

"""완성본 XXXXX, 임시 파일"""
class EWScheduler:
    """Error/Warning 데이터 API 스케줄러 클래스"""

    BASE_URL = "https://api-wmumpxg2lq-du.a.run.app/api/error-data"
    _ZERO_CODES = {"0x0000000000000000", "0x0", "0"}

    def __init__(self, db_session):
        """
        스케줄러 초기화

        Args:
            db_session: SQLAlchemy DB Session
        """
        self.db = db_session
        self.logger = logger

    def _parse_datetime(self, dt_str: any) -> datetime | None:
        """API 응답 시간 문자열을 datetime으로 변환"""
        if not dt_str:
            return None
        if isinstance(dt_str, str):
            return datetime.strptime(dt_str, "%Y%m%d%H%M%S")
        return dt_str

    def _find_device_by_identity(self, serial_no: str, device_type: int, device_num: int | str):
        """serial_no + device_type + device_num 조합으로 디바이스 조회"""
        return (
            self.db.query(TB_DEVICE)
            .filter(
                TB_DEVICE.serial_no == serial_no,
                TB_DEVICE.device_type == str(device_type),
                TB_DEVICE.device_num == str(device_num),
            )
            .first()
        )

    def _normalize_code(self, code: any) -> str | None:
        """에러/경보 코드 정규화 (유효하지 않으면 None)"""
        if code is None:
            return None
        code_str = str(code).strip()
        if not code_str or code_str.lower() in self._ZERO_CODES:
            return None
        return code_str

    def _find_existing_log(
        self, device_id, ew_dt: datetime, code: str, error_warn: int
    ) -> TB_WARN_ERROR_LOG | None:
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

    def _insert_ew_log(
        self,
        device_id,
        ew_dt: datetime,
        code: str,
        error_warn: int,
        ew_note: str | None = None,
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
        try:
            params = {
                "startDate": start_dt.strftime("%Y%m%d%H%M%S"),
                "endDate": end_dt.strftime("%Y%m%d%H%M%S"),
                "limit": limit,
            }

            response = requests.get(self.BASE_URL, params=params)

            if response.status_code == 200 and response.text.strip():
                data = response.json()
                self.logger.info(f"EW API 데이터 수신 성공: {len(data)}개 항목")
                return data

            self.logger.error(f"EW API 요청 실패: {response.status_code}")
            return []
        except Exception as e:
            self.logger.error(f"EW API 요청 중 에러 발생: {e}")
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
                error_note = item.get("errorNote") or item.get("error_note")
                warn_note = item.get("warnNote") or item.get("warn_note")

                if error_code:
                    inserted = self._insert_ew_log(
                        device_id=device.device_id,
                        ew_dt=ew_dt,
                        code=error_code,
                        error_warn=0,
                        ew_note=error_note,
                    )
                    if inserted:
                        success_count += 1

                if warn_code:
                    inserted = self._insert_ew_log(
                        device_id=device.device_id,
                        ew_dt=ew_dt,
                        code=warn_code,
                        error_warn=1,
                        ew_note=warn_note,
                    )
                    if inserted:
                        success_count += 1

            except Exception as e:
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