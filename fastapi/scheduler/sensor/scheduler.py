import os
import time
from datetime import datetime
from typing import Dict, Optional, Union

import requests
from dotenv import load_dotenv
from Logger import Logger as logger

from setting.database_orm import db_connection_pool
from db.public.models import *
from infrastructure.queryFactory.base_orm import BaseQueryFactory

load_dotenv()


class SensorScheduler:
    """센서 데이터 API 스케줄러 클래스"""

    # 데이터 조회 후 분류 용
    DEVICE_TYPE_AI_PEMS = 1
    DEVICE_TYPE_PEMSPRO = 2
    DEVICE_TYPE_PEMSPROPLUS = 3
    DEVICE_TYPE_FLOW = 4

    def __init__(self, db_session):
        """
        스케줄러 초기화

        Args:
            db_session: SQLAlchemy DB Session
        """
        self.db = db_session
        self.logger = logger
        self.base_url = os.getenv("KY_DEVICE_DATA_URL", "").strip()
        self.api_token = os.getenv("KY_API_TOKEN", "").strip()
        self.timeout_sec = float(os.getenv("KY_API_TIMEOUT", "10"))
        self.retries = int(os.getenv("KY_API_RETRIES", "2"))
        self.retry_backoff_sec = float(os.getenv("KY_API_RETRY_BACKOFF", "1.0"))
        self.http = requests.Session()
        self.http.headers.update({"Accept": "application/json"})
        if self.api_token:
            self.http.headers.update({"Authorization": f"Bearer {self.api_token}"})

    def _safe_rollback(self) -> None:
        """레코드 처리 중 DB 예외가 나면 다음 루프 전에 세션 상태를 복구한다."""
        try:
            self.db.rollback()
        except Exception as rollback_error:
            self.logger.error(f"센서 스케줄러 롤백 실패: {rollback_error}")

    def _get_or_create_device_by_identity(
        self, serial_no: str, device_type: int, device_num: Union[int, str], extra_data: Optional[Dict] = None
    ):
        """serial_no + device_type + device_num 기준으로 디바이스 조회/생성"""
        # 디바이스 고유 식별자(시리얼/타입/번호)로 기존 레코드 조회
        device = BaseQueryFactory(self.db, TB_DEVICE)._find_device_by_identity(serial_no, device_type, device_num)
        if device:
            if extra_data:
                device_query = BaseQueryFactory(self.db, TB_DEVICE)
                device = device_query.update(device, **extra_data)
            return device
        # 기존 디바이스가 없으면 신규 디바이스 기본 데이터 구성
        device_data = {
            'serial_no': serial_no,
            'device_type': str(device_type),
            'device_num': str(device_num),
        }
        # 추가 갱신 데이터가 있으면 신규 생성 데이터에도 합침
        if extra_data:
            device_data.update(extra_data)

        device_query = BaseQueryFactory(self.db, TB_DEVICE)
        # 신규 디바이스 레코드 생성
        return device_query.insert_single_row(**device_data)

    def fetch_api_data(self, start_dt: datetime, end_dt: datetime, limit: int = 50000) -> list:
        """
        API에서 센서 데이터 조회
        
        Args:
            start_dt: 시작 시간 (datetime)
            end_dt: 종료 시간 (datetime)
            limit: 조회 데이터 한계
            
        Returns:
            API 응답 데이터 리스트
        """
        # API 요구 포맷(YYYYMMDDHHMMSS)로 기간 파라미터 구성
        params = {
            "startDate": start_dt.strftime("%Y%m%d%H%M%S"),
            "endDate": end_dt.strftime("%Y%m%d%H%M%S"),
            "limit": limit,
        }

        if not self.base_url:
            self.logger.error("KY API 수신 실패: KY_DEVICE_DATA_URL 환경변수가 비어 있습니다")
            return []

        last_err = None

        for attempt in range(self.retries + 1):
            try:
                # GET 요청으로 센서 데이터 수집
                resp = self.http.get(self.base_url, params=params, timeout=self.timeout_sec)

                self.logger.info(f"KY API 요청: {resp.url}")

                # ---- 상태별 짧은 한글 로그 ----
                if resp.status_code == 200:
                    if not resp.text or not resp.text.strip():
                        self.logger.info("KY API 응답: 200 (데이터 없음)")
                        return []

                    try:
                        data = resp.json()
                    except ValueError as e:
                        preview = resp.text[:300].replace("\n", " ")
                        self.logger.error(f"KY API JSON 파싱 실패: {e} / body={preview}")
                        return []

                    if not isinstance(data, list):
                        self.logger.error(f"KY API 응답 형식 오류: list가 아니라 {type(data).__name__}")
                        return []

                    self.logger.info(f"KY API 응답: 200 (데이터 {len(data)}건)")
                    return data

                if resp.status_code == 404:
                    # URL/서비스 변경 의심
                    self.logger.error("KY API 응답: 404 (엔드포인트 없음) → Base URL/경로 변경 의심")
                    return []

                if resp.status_code in (401, 403):
                    self.logger.error(f"KY API 응답: {resp.status_code} (인증/권한 문제)")
                    return []

                # 기타 오류
                preview = resp.text[:300].replace("\n", " ")
                self.logger.error(f"KY API 응답: HTTP {resp.status_code} (요청 실패) / body={preview}")
                return []

            except requests.RequestException as e:
                last_err = e
                # 재시도 로그도 한글로 짧게
                if attempt < self.retries:
                    wait = self.retry_backoff_sec * (attempt + 1)
                    self.logger.warning(
                        f"KY API 통신 오류: {type(e).__name__} → {wait:.1f}초 후 재시도 ({attempt+1}/{self.retries})"
                    )
                    time.sleep(wait)
                    continue

                self.logger.error(f"KY API 최종 실패 (재시도 종료): {type(e).__name__}: {e}")
                return []
            except Exception as e:
                self.logger.error(f"KY API 수신 처리 중 예외 발생: {type(e).__name__}: {e}")
                return []

        # 보통 도달 안 함
        self.logger.error(f"KY API 최종 실패: {last_err}")
        return []

    def classify_data(self, data: list) -> tuple:
        """
        API 데이터를 디바이스 타입별로 분류
        
        Args:
            data: 원본 데이터 리스트
            
        Returns:
            (pemsproplus_data, flow_data, ai_pems_data) 튜플
             
        """
        # deviceType == PEMSPro 인 항목만 필터링
        pemspro_data = [
            item for item in data
            if item.get('deviceInfo', {}).get('deviceType') == self.DEVICE_TYPE_PEMSPRO
        ]
         # deviceType == AI_PEMS 인 항목만 필터링
        ai_pems_data = [
            item for item in data
            if item.get('deviceInfo', {}).get('deviceType') == self.DEVICE_TYPE_AI_PEMS
        ]
        # deviceType == PEMSPROPLUS 인 항목만 필터링
        pemsproplus_data = [
            item for item in data
            if item.get('deviceInfo', {}).get('deviceType') == self.DEVICE_TYPE_PEMSPROPLUS
        ]
        # deviceType == FLOW 인 항목만 필터링
        flow_data = [
            item for item in data
            if item.get('deviceInfo', {}).get('deviceType') == self.DEVICE_TYPE_FLOW
        ]

        self.logger.info(
            f"데이터 분류 완료 - AI_PEMS: {len(ai_pems_data)}, PEMSPRO: {len(pemspro_data)}, "
            f"PEMSPROPLUS: {len(pemsproplus_data)}, FLOW: {len(flow_data)}"
        )
        return ai_pems_data, pemspro_data, pemsproplus_data, flow_data

    def _parse_datetime(self, dt_str: any) -> datetime:
        """
        API 응답 시간 문자열을 datetime으로 변환
        
        Args:
            dt_str: 시간 문자열 (YYYYMMDDHHMMSS)
            
        Returns:
            datetime 객체
        """
        # 문자열일 경우 포맷에 맞춰 datetime 변환
        if isinstance(dt_str, str):
            return datetime.strptime(dt_str, '%Y%m%d%H%M%S')
        return dt_str

    def process_pemsproplus(self, pemsproplus_data: list) -> int:
        """
        PEMSPROPLUS 데이터 처리 및 적재
        Args:
            pemsproplus_data: PEMSPROPLUS 데이터 리스트
            
        Returns:
            성공적으로 적재된 데이터 개수
        """
        success_count = 0

        for item in pemsproplus_data:
            try:
                # 디바이스 식별 정보 추출
                serial_no = item['deviceInfo']['serialNo']
                device_num = item['deviceInfo']['deviceNum']
                device_type = item['deviceInfo']['deviceType']
                # 로그 시간 파싱
                log_dt = self._parse_datetime(item['dt'])

                # 디바이스 조회 또는 생성
                # 디바이스 갱신 데이터 구성 (device 테이블에 존재하는 컬럼만 반영)
                update_data = {
                    'cs_set_time': item['pemsProPlus'].get('csSetTime'),
                    'mg_refill_set_time': item['pemsProPlus'].get('mgRefillSetTime'),
                }
                # 디바이스 없으면 생성, 있으면 상태 업데이트
                device = self._get_or_create_device_by_identity(
                    serial_no, device_type, device_num, extra_data=update_data
                )
                device_id = device.device_id
                # 2. TB_PEMSPROPLUS_LOG 적재 (중복 확인)`
                pemsproplus_log_query = BaseQueryFactory(self.db, TB_PEMSPROPLUS_LOG)
                # 동일 device_id + log_dt 로그가 있으면 중복으로 간주
                existing_log = pemsproplus_log_query.find_one(device_id=device_id, log_dt=log_dt)

                if not existing_log:
                    # PEMSPROPLUS 센서/전원/환경 데이터를 테이블 스키마에 매핑
                    pemsproplus_log_data = {
                        'device_id': device_id,
                        'log_dt': log_dt,
                        'pressure': item.get('pemsProPlus', {}).get('pressure'),
                        'comp_temp': item.get('pemsProPlus', {}).get('temperature'),
                        'hz': item.get('pemsProPlus', {}).get('hz'),
                        'vsd_fsd': item.get('pemsProPlus', {}).get('dataType'),
                        'op_status': item.get('pemsProPlus', {}).get('opStatus'),
                        'avg_voltage': item.get('powerData', {}).get('avgVoltage'),
                        'avg_current': item.get('powerData', {}).get('avgCurrent'),
                        'cur_voltage': item.get('powerData', {}).get('curVoltage'),
                        'factor': item.get('powerData', {}).get('factor'),
                        'temperature': item.get('humidityData', {}).get('temperature'),
                        'humidity': item.get('humidityData', {}).get('humidity'),
                        'op_time': item.get('pemsProPlus', {}).get('opTime'),
                        'cs_usage_time': item.get('pemsProPlus', {}).get('csUsageTime'),
                        'mg_refill_time': item.get('pemsProPlus', {}).get('mgRefillTime'),
                    }
                    
                    # 신규 로그 적재
                    pemsproplus_log_query.insert_single_row(**pemsproplus_log_data)
                    self.logger.info(f"PEMSPROPLUS_LOG 적재 완료: {serial_no} ({log_dt})")
                    success_count += 1
                else:
                     # 이미 존재하는 로그는 적재하지 않음
                    self.logger.info(f"PEMSPROPLUS_LOG 중복 스킵: {serial_no} ({log_dt})")

                # 3. TB_VIBRATION_LOG 적재 (중복 확인)
                vibration_log_query = BaseQueryFactory(self.db, TB_VIBRATION_LOG)
                # 진동 로그도 동일 device_id + log_dt 기준으로 중복 확인
                existing_vibration = vibration_log_query.find_one(device_id=device_id, log_dt=log_dt)

                if not existing_vibration:
                    # 진동 데이터 필드 매핑
                    vibration_data = {
                        'device_id': device_id,
                        'frequency_1': item.get('vibrationData', {}).get('frequency1'),
                        'magnitude_1': item.get('vibrationData', {}).get('magnitude1'),
                        'frequency_2': item.get('vibrationData', {}).get('frequency2'),
                        'magnitude_2': item.get('vibrationData', {}).get('magnitude2'),
                        'frequency_3': item.get('vibrationData', {}).get('frequency3'),
                        'magnitude_3': item.get('vibrationData', {}).get('magnitude3'),
                    }

                    # 신규 진동 로그 적재
                    vibration_log_query.insert_single_row(**vibration_data)
                    self.logger.info(f"VIBRATION_LOG 적재 완료: {serial_no} ({log_dt})")
                else:
                    # 이미 존재하는 진동 로그는 적재하지 않음
                    self.logger.info(f"VIBRATION_LOG 중복 스킵: {serial_no} ({log_dt})")

            except Exception as e:
                # 특정 레코드 처리 실패 시 다음 레코드로 진행
                self._safe_rollback()
                self.logger.error(f"PEMSPROPLUS 데이터 적재 중 에러 발생: {e}")
                continue

        return success_count

    def process_pemspro(self, pemspro_data: list) -> int:
        """
        PERMPRO 데이터 처리 및 적재
        
        Args:
            pemspro_data: PERMPRO 데이터 리스트

        Returns:
            성공적으로 적재된 데이터 개수
        """
        success_count = 0

        for item in pemspro_data:
            try:
                serial_no = item['deviceInfo']['serialNo']
                device_num = item['deviceInfo']['deviceNum']
                device_type = item['deviceInfo']['deviceType']
                # 가능하면 로그 시간 파싱 (중복체크에 사용할 수 있음)
                log_dt = self._parse_datetime(item.get('dt')) if item.get('dt') else None

                # 1. 디바이스 조회 또는 생성
                device = self._get_or_create_device_by_identity(
                    serial_no, device_type, device_num
                )
                device_id = device.device_id

                # 2. PEMS_PRO 적재
                # 유연하게 필드 채우기: pemsPro 블록 또는 top-level에서 찾기
                pems_block = item.get('pemsPro') or item.get('pemsProPlus') or item
                power_block = item.get('powerData', {})

                pems_log_query = BaseQueryFactory(self.db, TB_PEMS_PRO_LOG)

                pems_log_data = {
                    'device_id': device_id,
                    'log_dt': log_dt,
                    'pressure': pems_block.get('pressure'),
                    'temperature': pems_block.get('temperature'),
                    'hz': pems_block.get('hz'),
                    'op_status': pems_block.get('opStatus'),
                    'avg_voltage': power_block.get('avgVoltage'),
                    'avg_current': power_block.get('avgCurrent'),
                    'cur_voltage': power_block.get('curVoltage'),
                    'factor': power_block.get('factor'),
                    'op_time': pems_block.get('opTime'),
                    'cs_usage_time': pems_block.get('csUsageTime'),
                    'mg_refill_time': pems_block.get('mgRefillTime'),
                }

                pems_log_query.insert_single_row(**pems_log_data)
                self.logger.info(f"PERMPRO(LOG) 적재 완료: {serial_no} ({log_dt})")
                success_count += 1

            except Exception as e:
                self._safe_rollback()
                self.logger.error(f"PERMPRO 데이터 적재 중 에러 발생: {e}")
                continue

        return success_count

    def process_flow(self, flow_data: list) -> int:
        """
        FLOW 데이터 처리 및 적재
                
        Args:
            flow_data: FLOW 데이터 리스트
            
        Returns:
            성공적으로 적재된 데이터 개수
        """
        success_count = 0

        for item in flow_data:
            try:
                # 디바이스 식별 정보 추출
                serial_no = item['deviceInfo']['serialNo']
                device_num = item['deviceInfo']['deviceNum']
                device_type = item['deviceInfo']['deviceType']
                # 로그 시간 파싱
                log_dt = self._parse_datetime(item['dt'])

                # 1. 디바이스 조회 또는 생성
                # FLOW는 상태 업데이트 없이 식별 정보만으로 조회/생성
                device = self._get_or_create_device_by_identity(
                    serial_no, device_type, device_num
                )
                device_id = device.device_id

                # 2. TB_FLOW_LOG 적재 (중복 확인)
                flow_log_query = BaseQueryFactory(self.db, TB_FLOW_LOG)
                # 동일 device_id + log_dt 로그가 있으면 중복으로 간주
                existing_flow_log = flow_log_query.find_one(device_id=device_id, log_dt=log_dt)

                if not existing_flow_log:
                    # FLOW 센서 데이터를 테이블 스키마에 매핑
                    flow_log_data = {
                        'device_id': device_id,
                        'log_dt': log_dt,
                        'pressure': item.get('flowData', {}).get('pressure'),
                        'temperature': item.get('flowData', {}).get('temperature'),
                        'cur_flow': item.get('flowData', {}).get('curFlow'),
                    }

                    # 신규 로그 적재
                    flow_log_query.insert_single_row(**flow_log_data)
                    self.logger.info(f"FLOW_LOG 적재 완료: {serial_no} ({log_dt})")
                    success_count += 1
                else:
                     # 이미 존재하는 로그는 적재하지 않음
                    self.logger.info(f"FLOW_LOG 중복 스킵: {serial_no} ({log_dt})")

            except Exception as e:
                # 특정 레코드 처리 실패 시 다음 레코드로 진행
                self._safe_rollback()
                self.logger.error(f"FLOW 데이터 적재 중 에러 발생: {e}")
                continue

        return success_count

    def process_ai_pems(self, ai_pems_data: list) -> int:
        """
        AI_PEMS 데이터 처리 및 적재
                
        Args:
            ai_pems_data: AI_PEMS 데이터 리스트
            
        Returns:
            성공적으로 적재된 데이터 개수
        """
        success_count = 0

        for item in ai_pems_data:
            try:
                # 디바이스 식별 정보 추출
                serial_no = item['deviceInfo']['serialNo']
                device_num = item['deviceInfo']['deviceNum']
                device_type = item['deviceInfo']['deviceType']
                # 로그 시간 파싱
                log_dt = self._parse_datetime(item['dt'])

                # 1. 디바이스 조회 또는 생성
                device = self._get_or_create_device_by_identity(
                    serial_no, device_type, device_num
                )
                device_id = device.device_id

                # 2. TB_AI_PEMS_LOG 적재 (중복 확인)
                ai_pems_log_query = BaseQueryFactory(self.db, TB_AI_PEMS_LOG)
                existing_log = ai_pems_log_query.find_one(device_id=device_id, log_dt=log_dt)

                if not existing_log:
                    # AI_PEMS 센서/전원 데이터를 테이블 스키마에 매핑
                    ai_pems_log_data = {
                        'device_id': device_id,
                        'log_dt': log_dt,
                        'pressure': item.get('aiPems', {}).get('pressure'),
                        'temperature': item.get('aiPems', {}).get('temperature'),
                        'hz': item.get('aiPems', {}).get('hz'),
                        'op_status': item.get('aiPems', {}).get('opStatus'),
                        'avg_voltage': item.get('powerData', {}).get('avgVoltage'),
                        'avg_current': item.get('powerData', {}).get('avgCurrent'),
                        'cur_voltage': item.get('powerData', {}).get('curVoltage'),
                        'factor': item.get('powerData', {}).get('factor'),
                        'op_time': item.get('aiPems', {}).get('opTime'),
                        'cs_usage_time': item.get('aiPems', {}).get('csUsageTime'),
                        'mg_refill_time': item.get('aiPems', {}).get('mgRefillTime'),
                    }
                    
                    # 신규 로그 적재
                    ai_pems_log_query.insert_single_row(**ai_pems_log_data)
                    self.logger.info(f"AI_PEMS_LOG 적재 완료: {serial_no} ({log_dt})")
                    success_count += 1
                else:
                    # 이미 존재하는 로그는 적재하지 않음
                    self.logger.info(f"AI_PEMS_LOG 중복 스킵: {serial_no} ({log_dt})")

            except Exception as e:
                # 특정 레코드 처리 실패 시 다음 레코드로 진행
                self._safe_rollback()
                self.logger.error(f"AI_PEMS 데이터 적재 중 에러 발생: {e}")
                continue

        return success_count

    def run(self, start_dt: datetime, end_dt: datetime):
        """
        스케줄러 실행
                
        Args:
            start_dt: 시작 시간 (datetime)
            end_dt: 종료 시간 (datetime)
        """
        self.logger.info("센서 스케줄러 시작")

        # 1. API 데이터 조회
        data = self.fetch_api_data(start_dt, end_dt)

        if not data:
            # 데이터가 없으면 즉시 종료
            self.logger.warning("처리할 데이터가 없습니다.")
            return

        # 2. 데이터 분류 (PERMPRO, PEMSPROPLUS, FLOW)
        ai_pems_data, pemspro_data, pemsproplus_data, flow_data = self.classify_data(data)

        pemspro_success = self.process_pemspro(pemspro_data)
        self.logger.info(f"PERMPRO 데이터 처리 완료: {pemspro_success}/{len(pemspro_data)}")

        ai_pems_success = self.process_ai_pems(ai_pems_data)
        self.logger.info(f"AI_PEMS 데이터 처리 완료: {ai_pems_success}/{len(ai_pems_data)}")

        pemsproplus_success = self.process_pemsproplus(pemsproplus_data)
        self.logger.info(f"PEMSPROPLUS 데이터 처리 완료: {pemsproplus_success}/{len(pemsproplus_data)}")

        flow_success = self.process_flow(flow_data)
        self.logger.info(f"FLOW 데이터 처리 완료: {flow_success}/{len(flow_data)}")

        # 전체 적재 결과 요약 로그
        total = pemspro_success + pemsproplus_success + flow_success
        self.logger.info(f"스케줄러 실행 완료 (총 {total}개 데이터 적재)")
