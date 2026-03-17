from sqlalchemy.orm import relationship
from db.base import Base
import uuid
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Boolean,
    Float,
    DateTime,
    ForeignKey,
    Text,
    BigInteger,
    Numeric,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, sessionmaker, declarative_base

# Base 클래스 생성

class TB_CUSTOMER(Base):
    __tablename__ = 'TB_CUSTOMER'

    customer_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, name='CUSTOMER_ID')
    customer_name = Column(Text, name='CUSTOMER_NAME', nullable=True)
    customer_industry = Column(String(50), name='CUSTOMER_INDUSTRY', nullable=True)
    customer_product = Column(String(50), name='CUSTOMER_PRODUCT', nullable=True)

    # Relationship (1:N with Device)
    devices = relationship("TB_DEVICE", back_populates="customer")
    peak_dispatch_runs = relationship("TB_PEAK_DISPATCH_RUN", back_populates="customer")

class TB_DEVICE(Base):
    __tablename__ = 'TB_DEVICE'

    device_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, name='DEVICE_ID')
    customer_id = Column(UUID(as_uuid=True), ForeignKey('TB_CUSTOMER.CUSTOMER_ID'), nullable=True, name='CUSTOMER_ID')
    serial_no = Column(Text, name='SERIAL_NO', nullable=True)
    device_type = Column(String(50), name='DEVICE_TYPE', nullable=True)
    device_num = Column(String(50), name='DEVICE_NUM', nullable=True)
    cs_set_time = Column(DateTime, name='CS_SET_TIME', nullable=True)
    mg_refill_set_time = Column(DateTime, name='MG_REFILL_SET_TIME', nullable=True)
    op_status = Column(Boolean, name='OP_STATUS', nullable=True, comment='0 : 정지, 1 : 운전')

    # Relationships
    customer = relationship("TB_CUSTOMER", back_populates="devices")
    vibration_logs = relationship("TB_VIBRATION_LOG", back_populates="device")
    flow_logs = relationship("TB_FLOW_LOG", back_populates="device")
    warn_error_logs = relationship("TB_WARN_ERROR_LOG", back_populates="device")
    pemsproplus_logs = relationship("TB_PEMSPROPLUS_LOG", back_populates="device")
    pems_pro_logs = relationship("TB_PEMS_PRO_LOG", back_populates="device")
    ai_pems_logs = relationship("TB_AI_PEMS_LOG", back_populates="device")
    simulation_logs = relationship("TB_SIMULATION_LOG", back_populates="device")
    peak_dispatch_results = relationship("TB_PEAK_DISPATCH_DEVICE_RESULT", back_populates="device")

class TB_VIBRATION_LOG(Base):
    __tablename__ = 'TB_VIBRATION_LOG'

    log_id = Column(BigInteger, primary_key=True, autoincrement=True, name='LOG_ID')
    device_id = Column(UUID(as_uuid=True), ForeignKey('TB_DEVICE.DEVICE_ID'), nullable=False, name='DEVICE_ID')
    frequency_1 = Column(Integer, name='FREQUENCY_1', nullable=True)
    magnitude_1 = Column(Float, name='MAGNITUDE_1', nullable=True)
    frequency_2 = Column(Integer, name='FREQUENCY_2', nullable=True)
    magnitude_2 = Column(Float, name='MAGNITUDE_2', nullable=True)
    frequency_3 = Column(Integer, name='FREQUENCY_3', nullable=True)
    magnitude_3 = Column(Float, name='MAGNITUDE_3', nullable=True)

    # Relationship
    device = relationship("TB_DEVICE", back_populates="vibration_logs")

class TB_FLOW_LOG(Base):
    __tablename__ = 'TB_FLOW_LOG'

    log_id = Column(BigInteger, primary_key=True, autoincrement=True, name='LOG_ID')
    device_id = Column(UUID(as_uuid=True), ForeignKey('TB_DEVICE.DEVICE_ID'), nullable=False, name='DEVICE_ID')
    log_dt = Column(DateTime, name='LOG_DT', nullable=True)
    pressure = Column(Float, name='PRESSURE', nullable=True)
    temperature = Column(Float, name='TEMPERATURE', nullable=True)
    cur_flow = Column(Float, name='CUR_FLOW', nullable=True)

    # Relationship
    device = relationship("TB_DEVICE", back_populates="flow_logs")

class TB_WARN_ERROR_LOG(Base):
    __tablename__ = 'TB_WARN_ERROR_LOG'

    id = Column(BigInteger, primary_key=True, autoincrement=True, name='ID')
    device_id = Column(UUID(as_uuid=True), ForeignKey('TB_DEVICE.DEVICE_ID'), nullable=False, name='DEVICE_ID')
    error_warn = Column(Boolean, name='ERROR_WARN', nullable=True, comment='0:경보, 1:고장')
    code = Column(Text, name='CODE', nullable=True)
    ew_note = Column(Text, name='EW_NOTE', nullable=True)
    ew_dt = Column(DateTime, name='EW_DT', nullable=True)

    # Relationship
    device = relationship("TB_DEVICE", back_populates="warn_error_logs")

class TB_PEMSPROPLUS_LOG(Base):
    __tablename__ = 'TB_PEMSPROPLUS_LOG'

    log_id = Column(BigInteger, primary_key=True, autoincrement=True, name='LOG_ID')
    device_id = Column(UUID(as_uuid=True), ForeignKey('TB_DEVICE.DEVICE_ID'), nullable=False, name='DEVICE_ID')
    log_dt = Column(DateTime, name='LOG_DT', nullable=True)
    pressure = Column(Float, name='PRESSURE', nullable=True)
    comp_temp = Column(Float, name='COMP_TEMP', nullable=True, comment='pemsProPlus')
    hz = Column(Integer, name='HZ', nullable=True)
    vsd_fsd = Column(Boolean, name='VSD_FSD', nullable=True, comment='0 : FSD, 1 : VSD')
    op_status = Column(Boolean, name='OP_STATUS', nullable=True, comment='0 : 정지, 1: 운전')
    avg_voltage = Column(Float, name='AVG_VOLTAGE', nullable=True)
    avg_current = Column(Float, name='AVG_CURRENT', nullable=True)
    cur_voltage = Column(Float, name='CUR_VOLTAGE', nullable=True)
    factor = Column(Float, name='FACTOR', nullable=True)
    temperature = Column(Float, name='TEMPERATURE', nullable=True, comment='습도계 데이터')
    humidity = Column(Float, name='HUMIDITY', nullable=True, comment='습도계 데이터')
    op_time = Column(Integer, name='OP_TIME', nullable=True)
    cs_usage_time = Column(Integer, name='CS_USAGE_TIME', nullable=True, comment='리셋 시 0으로 계산')
    mg_refill_time = Column(Integer, name='MG_REFILL_TIME', nullable=True, comment='리셋 시 0으로 계산')

    # Relationship
    device = relationship("TB_DEVICE", back_populates="pemsproplus_logs")

class TB_PEMS_PRO_LOG(Base):
    __tablename__ = 'TB_PEMS_PRO_LOG'

    log_id = Column(BigInteger, primary_key=True, autoincrement=True, name='LOG_ID')
    device_id = Column(UUID(as_uuid=True), ForeignKey('TB_DEVICE.DEVICE_ID'), nullable=False, name='DEVICE_ID')
    log_dt = Column(DateTime, name='LOG_DT', nullable=True)
    pressure = Column(Float, name='PRESSURE', nullable=True)
    temperature = Column(Float, name='TEMPERATURE', nullable=True)
    hz = Column(Float, name='HZ', nullable=True)
    op_status = Column(Boolean, name='OP_STATUS', nullable=True)
    avg_voltage = Column(Float, name='AVGVOLTAGE', nullable=True)
    avg_current = Column(Float, name='AVGCURRENT', nullable=True)
    cur_voltage = Column(Float, name='CURVOLTAGE', nullable=True)
    factor = Column(Float, name='FACTOR', nullable=True)
    op_time = Column(Integer, name='OP_TIME', nullable=True)
    cs_usage_time = Column(Integer, name='CSUSAGETIME', nullable=True)
    mg_refill_time = Column(Integer, name='MGREFILLTIME', nullable=True)

    # Relationship
    device = relationship("TB_DEVICE", back_populates="pems_pro_logs")
    
class TB_AI_PEMS_LOG(Base):
    __tablename__ = 'TB_AI_PEMS_LOG'
    
    log_id = Column(BigInteger, primary_key=True, autoincrement=True, name='LOG_ID')
    device_id = Column(UUID(as_uuid=True), ForeignKey('TB_DEVICE.DEVICE_ID'), nullable=False, name='DEVICE_ID')
    log_dt = Column(DateTime, name='LOG_DT', nullable=True)
    pressure = Column(Float, name='PRESSURE', nullable=True)
    temperature = Column(Float, name='TEMPERATURE', nullable=True)
    hz = Column(Float, name='HZ', nullable=True)
    op_status = Column(Boolean, name='OP_STATUS', nullable=True)
    avg_voltage = Column(Float, name='AVGVOLTAGE', nullable=True)
    avg_current = Column(Float, name='AVGCURRENT', nullable=True)
    cur_voltage = Column(Float, name='CURVOLTAGE', nullable=True)
    factor = Column(Float, name='FACTOR', nullable=True)
    op_time = Column(Integer, name='OP_TIME', nullable=True)
    cs_usage_time = Column(Integer, name='CSUSAGETIME', nullable=True)
    mg_refill_time = Column(Integer, name='MGREFILLTIME', nullable=True)
    
    device = relationship("TB_DEVICE", back_populates="ai_pems_logs")


class TB_SIMULATION_LOG(Base):
    __tablename__ = 'TB_SIMULATION_LOG'

    log_id = Column(BigInteger, primary_key=True, autoincrement=True, name='LOG_ID')
    device_id = Column(UUID(as_uuid=True), ForeignKey('TB_DEVICE.DEVICE_ID'), nullable=False, name='DEVICE_ID')
    baseline_pd_time = Column(DateTime, name='BASELINE_PD_TIME', nullable=False)
    search_time = Column(Integer, name='SEARCH_TIME', nullable=False)
    use_model = Column(String(50), name='USE_MODEL', nullable=False)
    available_feature = Column(Integer, name='AVAILABLE_FEATURE', nullable=False)
    change_feature = Column(Integer, name='CHANGE_FEATURE', nullable=True)
    result_value = Column(JSONB, name='RESULT_VALUE', nullable=True)
    change_column_info = Column(JSONB, name='CHANGE_COLUMN_INFO', nullable=True)
    feature_importance = Column(JSONB, name='FEATURE_IMPORTANCE', nullable=True)

    device = relationship("TB_DEVICE", back_populates="simulation_logs")


class TB_PEAK_DISPATCH_RUN(Base):
    __tablename__ = 'TB_PEAK_DISPATCH_RUN'

    peak_run_id = Column(BigInteger, primary_key=True, autoincrement=True, name='PEAK_RUN_ID')
    customer_id = Column(UUID(as_uuid=True), ForeignKey('TB_CUSTOMER.CUSTOMER_ID'), nullable=True, name='CUSTOMER_ID')
    status = Column(Text, name='STATUS', nullable=False)
    success = Column(Boolean, name='SUCCESS', nullable=False, default=False)
    message = Column(Text, name='MESSAGE', nullable=True)
    lookback_hours = Column(Integer, name='LOOKBACK_HOURS', nullable=False)
    top_k = Column(Integer, name='TOP_K', nullable=False)
    idle_op_status_threshold = Column(Numeric(6, 4), name='IDLE_OP_STATUS_THRESHOLD', nullable=False)
    force_exceed_demo = Column(Boolean, name='FORCE_EXCEED_DEMO', nullable=False, default=False)
    force_exceed_margin_ratio = Column(Numeric(6, 4), name='FORCE_EXCEED_MARGIN_RATIO', nullable=False)
    device_count = Column(Integer, name='DEVICE_COUNT', nullable=False)
    peak_15_before = Column(Numeric(12, 3), name='PEAK_15_BEFORE', nullable=False)
    peak_15_after = Column(Numeric(12, 3), name='PEAK_15_AFTER', nullable=False)
    peak_30_before = Column(Numeric(12, 3), name='PEAK_30_BEFORE', nullable=False)
    peak_30_after = Column(Numeric(12, 3), name='PEAK_30_AFTER', nullable=False)
    objective_peak_sum = Column(Numeric(14, 3), name='OBJECTIVE_PEAK_SUM', nullable=False)
    total_slack = Column(Numeric(14, 3), name='TOTAL_SLACK', nullable=False)
    donor_device_ids = Column(JSONB, name='DONOR_DEVICE_IDS', nullable=True)
    idle_device_ids = Column(JSONB, name='IDLE_DEVICE_IDS', nullable=True)
    allocation_plan = Column(JSONB, name='ALLOCATION_PLAN', nullable=True)
    created_at = Column(DateTime, name='CREATED_AT', nullable=False)

    customer = relationship("TB_CUSTOMER", back_populates="peak_dispatch_runs")
    device_results = relationship("TB_PEAK_DISPATCH_DEVICE_RESULT", back_populates="peak_dispatch_run", cascade="all, delete-orphan")


class TB_PEAK_DISPATCH_DEVICE_RESULT(Base):
    __tablename__ = 'TB_PEAK_DISPATCH_DEVICE_RESULT'

    result_id = Column(BigInteger, primary_key=True, autoincrement=True, name='RESULT_ID')
    peak_run_id = Column(BigInteger, ForeignKey('TB_PEAK_DISPATCH_RUN.PEAK_RUN_ID', ondelete='CASCADE'), nullable=False, name='PEAK_RUN_ID')
    device_id = Column(UUID(as_uuid=True), ForeignKey('TB_DEVICE.DEVICE_ID'), nullable=False, name='DEVICE_ID')
    is_donor = Column(Boolean, name='IS_DONOR', nullable=False)
    is_idle = Column(Boolean, name='IS_IDLE', nullable=False)
    op_status_mean = Column(Numeric(8, 5), name='OP_STATUS_MEAN', nullable=False)
    threshold = Column(Numeric(12, 3), name='THRESHOLD', nullable=False)
    baseline_15 = Column(Numeric(12, 3), name='BASELINE_15', nullable=False)
    baseline_30 = Column(Numeric(12, 3), name='BASELINE_30', nullable=False)
    optimized_15 = Column(Numeric(12, 3), name='OPTIMIZED_15', nullable=False)
    optimized_30 = Column(Numeric(12, 3), name='OPTIMIZED_30', nullable=False)
    delta_15 = Column(Numeric(12, 3), name='DELTA_15', nullable=False)
    delta_30 = Column(Numeric(12, 3), name='DELTA_30', nullable=False)
    shift_in_15 = Column(Numeric(12, 3), name='SHIFT_IN_15', nullable=False)
    shift_in_30 = Column(Numeric(12, 3), name='SHIFT_IN_30', nullable=False)
    shift_out_15 = Column(Numeric(12, 3), name='SHIFT_OUT_15', nullable=False)
    shift_out_30 = Column(Numeric(12, 3), name='SHIFT_OUT_30', nullable=False)
    required_shift_15 = Column(Numeric(12, 3), name='REQUIRED_SHIFT_15', nullable=False)
    required_shift_30 = Column(Numeric(12, 3), name='REQUIRED_SHIFT_30', nullable=False)
    slack_15 = Column(Numeric(12, 3), name='SLACK_15', nullable=False)
    slack_30 = Column(Numeric(12, 3), name='SLACK_30', nullable=False)
    distributed_targets_15 = Column(JSONB, name='DISTRIBUTED_TARGETS_15', nullable=True)
    distributed_targets_30 = Column(JSONB, name='DISTRIBUTED_TARGETS_30', nullable=True)
    distribution_text = Column(Text, name='DISTRIBUTION_TEXT', nullable=True)
    created_at = Column(DateTime, name='CREATED_AT', nullable=False)

    peak_dispatch_run = relationship("TB_PEAK_DISPATCH_RUN", back_populates="device_results")
    device = relationship("TB_DEVICE", back_populates="peak_dispatch_results")
    