"""EW(에러/경보) 코드 매핑 유틸리티.

- 정수/문자열(0x 포함) 코드를 받아 비트마스크를 해석한다.
- 정의된 코드 설명 목록을 반환하며, 미정의 비트는 별도 표기한다.
"""

from typing import List, Optional, Union

# EW 코드 비트값과 설명 매핑. 나중에 DB로 옮기는게 좋아 보입니다 실제 운영시에
EW_CODE_MAP = {
    0x0000000000000001: "미확인 고장",
    0x0000000000000002: "비상 정지",
    0x0000000000000004: "EXT 1 통신 불량",
    0x0000000000000008: "EXT 2 통신 불량",
    0x0000000000000010: "EXT 1 이상",
    0x0000000000000020: "EXT 2 이상",
    0x0000000000000040: "인버터 통신 불량",
    0x0000000000000080: "인버터 이상",
    0x0000000000000100: "메인 모터 과전류",
    0x0000000000000200: "팬 모터 과전류",
    0x0000000000000400: "메인 모터 역상",
    0x0000000000000800: "냉각수 부족",
    0x0000000000001000: "메인 모터 쿨링팬 이상",
    0x0000000000002000: "온도센서 단선[T1]",
    0x0000000000004000: "온도센서 단선[T2]",
    0x0000000000008000: "미확인 고장",
    0x0000000000010000: "미확인 고장",
    0x0000000000020000: "온도센서 단선[T5]",
    0x0000000000040000: "압력센서 단선[P1]",
    0x0000000000080000: "압력센서 단선[P2]",
    0x0000000000100000: "압력센서 단선[P3]",
    0x0000000000200000: "오일 펌프 과전류",
    0x0000000000400000: "토출 공기 온도 상승 / 2단 토출 공기 온도 상승",
    0x0000000000800000: "오일 온도 상승",
    0x0000000001000000: "1단 토출 공기 온도 상승",
    0x0000000002000000: "에어 압력 상승 / 2단 토출 압력 상승",
    0x0000000004000000: "1단 토출 압력 상승",
    0x0000000008000000: "오일 압력 상승",
    0x0000000010000000: "오일 압력 저하",
    0x0000000020000000: "순간 정전",
    0x0000000040000000: "기동반 이상",
    0x0000000080000000: "압축기 모릴 점검",
    0x0000000100000000: "무부하 운전 과다 이상",
    0x0000000200000000: "인버터 4 ~ 20 mA 불량 F",
    0x0000000400000000: "인버터 4 ~ 20 mA 불량 L",
    0x0000000800000000: "DAC 통신 불량",
    0x0000001000000000: "인버터 이상 [저전압]",
    0x0000002000000000: "모터/스크류 1 진동 이상",
    0x0000004000000000: "모터/스크류 2 진동 이상",
}


def parse_ew_code(code: Union[int, str, None]) -> Optional[int]:
    """
    args:
        code: 정수 또는 16진수 문자열 형태의 EW 코드. None이나 빈 문자열, 0, 음수 등은 None으로 간주한다.
    returns:
        유효한 정수 코드 값. 0 또는 음수, 파싱 실패 시 None을 반환한다.
    """
    if code is None:
        return None

    if isinstance(code, int):
        return code if code > 0 else None

    code_text = str(code).strip()
    if not code_text:
        return None

    try:
        if code_text.lower().startswith("0x"):
            value = int(code_text, 16)
        else:
            value = int(code_text)
    except ValueError:
        return None

    return value if value > 0 else None


def decode_ew_code(code: Union[int, str, None]) -> List[str]:
    """
    args:
        code: 정수 또는 16진수 문자열 형태의 EW 코드. None이나 빈 문자열, 0, 음수 등은 None으로 간주한다.
    returns:
        코드에 매핑되는 설명 문자열 목록. 유효한 코드가 없으면 빈 리스트를 반환한다.
    """
    
    value = parse_ew_code(code)
    if value is None:
        return []

    labels: List[str] = []
    unknown_bits = value

    for bit in sorted(EW_CODE_MAP.keys()):
        if value & bit:
            labels.append(EW_CODE_MAP[bit])
            unknown_bits &= ~bit

    if unknown_bits:
        labels.append(f"미정의 코드(0x{unknown_bits:016X})")

    return labels


def decode_ew_code_text(code: Union[int, str, None], sep: str = ", ") -> str:
    """
    args:
        code: 정수 또는 16진수 문자열 형태의 EW 코드. None이나 빈 문자열, 0, 음수 등은 None으로 간주한다.
        sep: 여러 설명을 구분할 때 사용할 구분자 문자열. 기본값은 ", "입니다.
    returns:
        코드에 매핑되는 설명 문자열을 sep로 구분하여 반환한다. 유효한 코드가 없으면 빈 문자열을 반환한다.
    """
    
    labels = decode_ew_code(code)
    return sep.join(labels)
