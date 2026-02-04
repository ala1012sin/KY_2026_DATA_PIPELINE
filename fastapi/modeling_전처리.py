import numpy as np
import pandas as pd
import random
import statsmodels.api as sm
from math import sqrt
from scipy.stats import ks_2samp
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


def filter_double_cut(df, time_col='DATE', unit_col='UNIT', value_col='INST_POWER'):
    """
    유닛 필터링(2단계):
    1) 0 비율(Zero Ratio)이 90%를 초과하는 유닛 제거
    2) 최대 연속 가동 시간(Max Active Run)이 120 미만인 유닛 제거

    Parameters
    ----------
    df : pandas.DataFrame
        원본 데이터
    time_col : str
        시간 컬럼명
    unit_col : str
        유닛 식별 컬럼명
    value_col : str
        분석 대상 값 컬럼명(전력 등)

    Returns
    -------
    pandas.DataFrame
        필터링된 유닛만 포함한 데이터
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    unit_stats = []

    # 유닛 단위로 0 비율과 연속 가동 시간 계산
    for unit, group in df.groupby(unit_col):
        group = group.sort_values(time_col)
        total_count = len(group)

        # 0(또는 매우 작은 값) 비율 계산
        is_zero = group[value_col].abs() < 1e-3
        zero_ratio = is_zero.sum() / total_count if total_count > 0 else 1.0

        # 연속 가동(True) 구간의 최대 길이 계산
        is_active = ~is_zero
        if is_active.sum() == 0:
            max_active_run = 0
        else:
            run_groups = (is_active != is_active.shift()).cumsum()
            active_runs = group[is_active].groupby(run_groups).size()
            max_active_run = active_runs.max() if not active_runs.empty else 0

        unit_stats.append(
            {
                'unit_id': unit,
                'zero_ratio': zero_ratio,
                'max_active_run': max_active_run,
            }
        )

    stats_df = pd.DataFrame(unit_stats)

    # 1단계: 0 비율 기준 필터링
    survivors_step1 = stats_df[stats_df['zero_ratio'] <= 0.90].copy()

    # 2단계: 연속 가동 시간 기준 필터링
    final_survivors = survivors_step1[survivors_step1['max_active_run'] >= 120].copy()

    valid_unit_ids = final_survivors['unit_id'].tolist()
    return df[df[unit_col].isin(valid_unit_ids)].copy()


def resample_fleet_data(df, time_col='DATE', unit_col='UNIT', freq='1min'):
    """
    유닛별 시계열을 지정된 주기(freq)로 리샘플링하고,
    누락 구간은 NaN으로 남겨 결측을 명시합니다.

    Parameters
    ----------
    df : pandas.DataFrame
        원본 데이터
    time_col : str
        시간 컬럼명
    unit_col : str
        유닛 식별 컬럼명
    freq : str
        리샘플링 주기(예: '1min')

    Returns
    -------
    pandas.DataFrame
        리샘플링된 데이터
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    resampled_list = []

    # 유닛별로 인덱스를 시간으로 맞춘 뒤 리샘플링
    for unit, group in df.groupby(unit_col):
        group = group.set_index(time_col).sort_index()
        # 숫자 컬럼만 평균 집계(결측은 NaN 유지)
        resampled = group.resample(rule=freq).mean(numeric_only=True)
        # 리샘플링 과정에서 사라진 유닛 컬럼 복원
        resampled[unit_col] = unit
        resampled_list.append(resampled)

    final_df = pd.concat(resampled_list).reset_index()

    # 컬럼 순서 정리(시간, 유닛, 나머지)
    cols = [time_col, unit_col] + [c for c in final_df.columns if c not in [time_col, unit_col]]
    return final_df[cols]


def analyze_gap_statistics_numerical(df, unit_col='UNIT', time_col='DATE', value_col='INST_POWER'):
    """
    결측 구간(Gap) 길이를 수치적으로 분석합니다.

    Returns
    -------
    coverage_df : pandas.DataFrame
        제안 limit별 커버리지(몇 %의 gap을 덮는지)
    unit_stats_clean : pandas.DataFrame
        유닛별 gap 길이 분포(퍼센타일 포함)
    """
    gap_data = []

    # 유닛별 결측 구간 길이 수집
    for unit, group in tqdm(df.groupby(unit_col)):
        group = group.sort_values(time_col)
        is_na = group[value_col].isna()
        if is_na.sum() == 0:
            continue

        gap_groups = (is_na != is_na.shift()).cumsum()
        gaps = group[is_na].groupby(gap_groups).size()

        if not gaps.empty:
            for g in gaps:
                gap_data.append({'unit_id': unit, 'gap_size': int(g)})

    gap_df = pd.DataFrame(gap_data)
    if gap_df.empty:
        return None, None

    # limit 제안값별 커버리지 계산
    checkpoints = [5, 10, 20, 30, 45, 60, 90, 120, 180, 360]
    coverage_data = []
    total_gaps = len(gap_df)

    for cp in checkpoints:
        covered_count = (gap_df['gap_size'] <= cp).sum()
        coverage_pct = (covered_count / total_gaps) * 100
        coverage_data.append(
            {
                'Limit_Proposal (min)': cp,
                'Covered_Count': covered_count,
                'Coverage_Rate (%)': round(coverage_pct, 2),
                'Remaining_Gaps': total_gaps - covered_count,
            }
        )

    coverage_df = pd.DataFrame(coverage_data)

    # 유닛별 통계(퍼센타일 포함)
    unit_stats = gap_df.groupby('unit_id')['gap_size'].describe(percentiles=[0.5, 0.75, 0.90, 0.95, 0.99])
    cols = ['count', 'mean', '50%', '90%', '95%', '99%', 'max']
    unit_stats_clean = unit_stats[cols].round(1)

    return coverage_df, unit_stats_clean


def analyze_sensitivity_full_inspection(df, unit_col='UNIT', value_col='INST_POWER'):
    """
    민감도 분석(전수 조사 기반):
    - ACF 기반으로 limit 후보를 계산
    - 구간별 결측을 인위적으로 만들어 보간 정확도를 측정
    - RMSE / Peak Error / KS 통계로 평가

    Returns
    -------
    pandas.DataFrame or None
        유닛별 threshold/limit 및 평가 지표
    """
    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')

    thresholds = [0.9, 0.8, 0.7, 0.6, 0.5]
    final_results = []

    for unit in tqdm(df[unit_col].unique()):
        group = df[df[unit_col] == unit]

        # NaN이 아닌 연속 구간 중 길이 120 이상만 사용
        is_na = group[value_col].isna()
        is_active = ~is_na
        run_groups = (is_active != is_active.shift()).cumsum()
        valid_runs = [run[value_col].values for _, run in group[is_active].groupby(run_groups) if len(run) >= 120]

        if not valid_runs:
            continue

        # 유닛 전체 구간을 합쳐 ACF 계산(글로벌 ACF)
        all_data = np.concatenate(valid_runs)
        if len(all_data) > 120:
            acf_values = sm.tsa.acf(all_data, nlags=120, fft=True)
        else:
            acf_values = None

        for th in thresholds:
            # ACF가 threshold 아래로 떨어지는 지점의 첫 인덱스를 limit으로 사용
            if acf_values is not None:
                drops = np.where(acf_values < th)[0]
                limit = drops[0] if len(drops) > 0 else 60
            else:
                limit = 30

            if limit < 1:
                limit = 1

            metrics = {'rmse': [], 'peak_err': [], 'ks_stat': []}

            # 각 구간에서 중앙 일부를 결측으로 만들고 보간 성능 평가
            for segment in valid_runs:
                if len(segment) <= limit + 10:
                    continue

                start = (len(segment) - limit) // 2
                end = start + limit
                original = segment[start:end].copy()

                try:
                    sim_series = pd.Series(segment.copy())
                    sim_series.iloc[start:end] = np.nan
                    filled = sim_series.interpolate(method='linear', limit_direction='both')
                    pred = filled.iloc[start:end].values

                    if np.isnan(pred).any():
                        continue

                    # 1) RMSE
                    rmse = sqrt(mean_squared_error(original, pred))
                    metrics['rmse'].append(rmse)

                    # 2) Peak Error (최대값 차이)
                    peak_err = abs(original.max() - pred.max())
                    metrics['peak_err'].append(peak_err)

                    # 3) KS 통계량
                    ks, _ = ks_2samp(original, pred)
                    metrics['ks_stat'].append(ks)
                except Exception:
                    continue

            if metrics['rmse']:
                final_results.append(
                    {
                        'unit_id': unit,
                        'threshold': th,
                        'limit_minutes': int(limit),
                        'RMSE': np.mean(metrics['rmse']),
                        'Peak_Err': np.mean(metrics['peak_err']),
                        'KS_Stat': np.mean(metrics['ks_stat']),
                    }
                )

    results_df = pd.DataFrame(final_results)
    if results_df.empty:
        return None

    return results_df


def recommend_best_thresholds_multimetric(results_df):
    """
    유닛별 best threshold/limit를 선택합니다.
    - RMSE/Peak/KS를 각각 랭킹화
    - 랭킹 합계가 가장 낮은 조합을 최종 선택

    Returns
    -------
    pandas.DataFrame
        유닛별 추천 threshold/limit
    """
    recommendations = []

    for unit, group in results_df.groupby('unit_id'):
        group = group.copy()
        group['Rank_RMSE'] = group['RMSE'].rank(ascending=True)
        group['Rank_Peak'] = group['Peak_Err'].rank(ascending=True)
        group['Rank_KS'] = group['KS_Stat'].rank(ascending=True)
        group['Total_Score'] = group['Rank_RMSE'] + group['Rank_Peak'] + group['Rank_KS']

        # 동점이면 limit이 큰 쪽을 우선
        best_row = group.sort_values(['Total_Score', 'limit_minutes'], ascending=[True, False]).iloc[0]

        recommendations.append(
            {
                'unit_id': unit,
                'best_threshold': best_row['threshold'],
                'limit_minutes': int(best_row['limit_minutes']),
                'Total_Score': best_row['Total_Score'],
                'RMSE': round(best_row['RMSE'], 2),
                'Peak_Err': round(best_row['Peak_Err'], 2),
            }
        )

    return pd.DataFrame(recommendations)


def auto_compare_and_decide_limits(sensitivity_best_df, gap_stats_df):
    """
    ACF 기반 limit과 gap 통계를 비교하여 최종 limit을 결정합니다.

    Returns
    -------
    pandas.DataFrame
        유닛별 최종 limit과 상태(status)
    """
    if gap_stats_df.index.name == 'unit_id':
        gap_stats_df = gap_stats_df.reset_index()

    merged = pd.merge(
        sensitivity_best_df[['unit_id', 'limit_minutes', 'Total_Score']],
        gap_stats_df[['unit_id', '50%', '95%', '99%', 'max']],
        on='unit_id',
        how='left',
    )

    final_decisions = []
    for _, row in merged.iterrows():
        acf_limit = row['limit_minutes']
        real_99 = row['99%']
        real_max = row['max']

        # 상태 값은 단순 분류용(외부에서 재정의 가능)
        if acf_limit >= real_99:
            status = 'perfect'
            final_limit = acf_limit
        elif acf_limit >= row['95%']:
            status = 'good'
            final_limit = acf_limit
        else:
            status = 'data_loss'
            final_limit = acf_limit

        final_decisions.append(
            {
                'unit_id': row['unit_id'],
                'ACF_Limit (Theory)': int(acf_limit),
                'Gap_99% (Reality)': real_99,
                'Gap_Max': real_max,
                'Final_Limit': int(final_limit),
                'Status': status,
            }
        )

    return pd.DataFrame(final_decisions)


def compare_interpolation_multimetric(df, unit_col='UNIT', value_col='INST_POWER'):
    """
    여러 보간법(Linear/Akima/Pchip/Ffill)을 비교 평가합니다.
    - RMSE / Peak Error / KS 통계 기준으로 유닛별 우수 방법을 선택

    Returns
    -------
    score_df : pandas.DataFrame
        모든 방법의 점수
    best_methods : pandas.DataFrame
        유닛별 최적 방법
    """
    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')

    random.seed(42)
    np.random.seed(42)

    methods = [
        {'name': 'Linear', 'method': 'linear', 'order': None},
        {'name': 'Akima', 'method': 'akima', 'order': None},
        {'name': 'Pchip', 'method': 'pchip', 'order': None},
        {'name': 'Ffill', 'method': 'ffill', 'order': None},
    ]

    results = []

    for unit in tqdm(df[unit_col].unique()):
        group = df[df[unit_col] == unit]

        # 연속 구간만 추출(길이 60 이상)
        is_active = group[value_col].notna()
        run_groups = (is_active != is_active.shift()).cumsum()
        valid_runs = [run[value_col].values for _, run in group[is_active].groupby(run_groups) if len(run) >= 60]

        if not valid_runs:
            continue

        # 평가용 구간 목록
        test_segments = valid_runs
        gap = 5

        for method_info in methods:
            metrics = {'rmse': [], 'peak_err': [], 'ks_stat': []}

            for segment in test_segments:
                if len(segment) <= gap + 10:
                    continue

                # 중앙부를 gap으로 설정하여 보간 테스트
                start = (len(segment) - gap) // 2
                end = start + gap
                original = segment[start:end].copy()

                try:
                    sim_series = pd.Series(segment.copy())
                    sim_series.iloc[start:end] = np.nan

                    if method_info['method'] == 'ffill':
                        filled = sim_series.ffill()
                    else:
                        filled = sim_series.interpolate(
                            method=method_info['method'],
                            order=method_info['order'],
                            limit_direction='both',
                        )

                    pred = filled.iloc[start:end].values
                    if np.isnan(pred).any():
                        continue

                    rmse = sqrt(mean_squared_error(original, pred))
                    metrics['rmse'].append(rmse)

                    peak_diff = abs(original.max() - pred.max())
                    metrics['peak_err'].append(peak_diff)

                    ks_stat, _ = ks_2samp(original, pred)
                    metrics['ks_stat'].append(ks_stat)
                except Exception:
                    continue

            if metrics['rmse']:
                results.append(
                    {
                        'unit_id': unit,
                        'Method': method_info['name'],
                        'RMSE': round(np.mean(metrics['rmse']), 4),
                        'Peak_Err': round(np.mean(metrics['peak_err']), 4),
                        'KS_Stat': round(np.mean(metrics['ks_stat']), 4),
                    }
                )

    score_df = pd.DataFrame(results)
    if score_df.empty:
        return None, None

    # 지표별 랭킹 계산
    score_df['Rank_RMSE'] = score_df.groupby('unit_id')['RMSE'].rank(ascending=True)
    score_df['Rank_Peak'] = score_df.groupby('unit_id')['Peak_Err'].rank(ascending=True)
    score_df['Rank_KS'] = score_df.groupby('unit_id')['KS_Stat'].rank(ascending=True)

    # 랭킹 합계가 낮을수록 우수
    score_df['Total_Score'] = score_df['Rank_RMSE'] + score_df['Rank_Peak'] + score_df['Rank_KS']

    best_methods = score_df.sort_values('Total_Score').groupby('unit_id').head(1)
    return score_df, best_methods


def generate_final_dataset_all_vars(
    df,
    limit_settings,
    method_settings,
    time_col='DATE',
    unit_col='UNIT',
    target_col='INST_POWER',
):
    """
    유닛별 limit과 method를 적용하여 모든 수치형 변수에 보간을 수행합니다.
    - limit 이하의 결측 구간만 보간
    - target_col 기준으로 session_id 재구성

    Returns
    -------
    pandas.DataFrame
        최종 전처리된 데이터
    """
    df = df.copy()
    df[unit_col] = df[unit_col].astype(str)

    # 유닛별 limit 매핑 생성
    limit_map = {}
    if limit_settings is not None:
        limit_settings = limit_settings.copy()
        if 'unit_id' in limit_settings.columns:
            limit_settings['unit_id'] = limit_settings['unit_id'].astype(str)
            settings_col = 'unit_id'
        else:
            settings_col = unit_col
        limit_map = limit_settings.set_index(settings_col)['limit_minutes'].to_dict()

    # 유닛별 method 매핑 생성
    method_map = {}
    if method_settings is not None:
        winners_col = 'unit_id' if 'unit_id' in method_settings.columns else unit_col
        for _, row in method_settings.iterrows():
            method_name = row['Method']
            if 'Akima' in method_name:
                config = {'method': 'akima', 'order': None}
            elif 'Pchip' in method_name:
                config = {'method': 'pchip', 'order': None}
            elif 'Ffill' in method_name:
                config = {'method': 'ffill', 'order': None}
            else:
                config = {'method': 'linear', 'order': None}
            method_map[row[winners_col]] = config

    # 보간 대상 수치형 컬럼 선택
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_interpolate = [c for c in numeric_cols if c not in [unit_col, 'unit_id', 'session_id']]

    processed_list = []

    for unit, group in tqdm(df.groupby(unit_col)):
        limit = limit_map.get(unit, 60)
        method_conf = method_map.get(unit, {'method': 'linear', 'order': None})

        group = group.sort_values(time_col).copy()

        # 컬럼별 보간 처리
        for col in cols_to_interpolate:
            is_na = group[col].isna()
            if is_na.sum() == 0:
                continue

            # 연속 결측 구간 크기 계산
            gap_groups = (is_na != is_na.shift()).cumsum()
            gap_sizes = group.groupby(gap_groups)[col].transform('size')

            # limit 이하 구간만 보간 대상으로 지정
            mask_fillable = is_na & (gap_sizes <= limit)

            try:
                if method_conf['method'] == 'ffill':
                    interpolated = group[col].ffill()
                else:
                    interpolated = group[col].interpolate(
                        method=method_conf['method'],
                        order=method_conf['order'],
                        limit_direction='both',
                    )
            except Exception:
                # 실패 시 선형 보간으로 fallback
                interpolated = group[col].interpolate(method='linear')

            # 원래 값이 존재하거나 보간 대상인 경우만 채움
            final_values = np.where((group[col].notna()) | mask_fillable, interpolated, np.nan)
            group[col] = final_values

        # target_col 기준으로 세션 분할
        still_na = group[target_col].isna()
        session_change = still_na != still_na.shift()
        session_ids = session_change.cumsum()
        group['session_id'] = group[unit_col].astype(str) + '_' + session_ids.astype(str)

        # target_col이 NaN인 행 제거 후 이동평균 생성
        group_clean = group.dropna(subset=[target_col]).copy()
        if len(group_clean) > 0:
            group_clean['MA_5'] = group_clean[target_col].rolling(window=5).mean()
            group_clean['MA_10'] = group_clean[target_col].rolling(window=10).mean()
            group_clean = group_clean.dropna()
            processed_list.append(group_clean)

    if not processed_list:
        return pd.DataFrame()

    return pd.concat(processed_list, ignore_index=True)


def clean_zero_shutdowns(df, unit_col='UNIT', time_col='DATE', value_col='INST_POWER', duration_limit=60):
    """
    전력이 0인 구간이 일정 시간(duration_limit) 이상 지속되면
    해당 구간을 제거하고 session_id를 재발급합니다.

    Returns
    -------
    pandas.DataFrame
        정리된 데이터
    """
    df = df.copy()

    # 0 구간 탐지 및 길이 계산
    is_zero = df[value_col] <= 1e-5
    zero_groups = (is_zero != is_zero.shift()).cumsum()
    group_sizes = df.groupby(zero_groups)[value_col].transform('size')

    # 제거 대상 마스크 생성
    mask_to_remove = is_zero & (group_sizes >= duration_limit)
    df_clean = df[~mask_to_remove].copy()

    if len(df_clean) == len(df):
        return df

    # 제거 후 세션 재구성
    final_list = []
    for unit, group in tqdm(df_clean.groupby(unit_col)):
        group = group.sort_values(time_col)

        time_diff = group[time_col].diff().dt.total_seconds() / 60.0
        is_break = time_diff > 1.5
        is_break.iloc[0] = False

        new_session_ids = is_break.cumsum()
        group['session_id'] = group[unit_col].astype(str) + '_new_' + new_session_ids.astype(str)

        final_list.append(group)

    return pd.concat(final_list, ignore_index=True)
