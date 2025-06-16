# -*- coding: utf-8 -*-

# 한글 자모 리스트 정의
CHOSEONG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JUNGSEONG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ',
                  'ㅣ']
JONGSEONG_LIST = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
                  'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']  # 첫 번째는 종성 없음

# 유니코드 한글 시작점 및 자모 개수
SBASE = 0xAC00  # 한글 음절 시작 유니코드 '가'
N_JUNGSEONG = 21  # 중성 개수
N_JONGSEONG = 28  # 종성 개수 (종성 없음 포함)


def _is_choseong(char):
    """주어진 문자가 초성인지 확인"""
    return char in CHOSEONG_LIST


def _is_jungseong(char):
    """주어진 문자가 중성인지 확인"""
    return char in JUNGSEONG_LIST


def _is_jongseong(char):
    """주어진 문자가 종성으로 사용될 수 있는지 확인 (종성 없음 제외)"""
    return char in JONGSEONG_LIST and JONGSEONG_LIST.index(char) != 0


def _compose_syllable(l, v, t=None):
    """초성, 중성, (선택적) 종성을 받아 한글 한 글자로 조합"""
    if not (_is_choseong(l) and _is_jungseong(v)):
        # 유효하지 않은 초성 또는 중성이면 조합 불가
        # 이 경우는 보통 발생하지 않도록 입력이 제어되어야 함
        parts = [part for part in [l, v, t] if part]
        return "".join(parts)

    l_idx = CHOSEONG_LIST.index(l)
    v_idx = JUNGSEONG_LIST.index(v)
    t_idx = 0  # 기본값은 종성 없음
    if t and t in JONGSEONG_LIST:
        t_idx = JONGSEONG_LIST.index(t)

    syllable_code = SBASE + (l_idx * N_JUNGSEONG + v_idx) * N_JONGSEONG + t_idx
    return chr(syllable_code)


def combine_hangul_jamo(jamo_sequence):
    """
    한글 자모음 시퀀스를 입력받아 완성된 한글 문자열로 조합합니다.
    예: ['ㄱ', 'ㅏ', 'ㄴ', 'ㅣ'] -> "가니"
    예: ['ㅎ', 'ㅏ', 'ㄴ', 'ㄱ', 'ㅡ', 'ㄹ'] -> "한글"
    """
    result = []
    syllable_buffer = []  # 현재 조합 중인 음절의 자모 [초성?, 중성?, 종성?]

    for i, current_jamo in enumerate(jamo_sequence):
        if not syllable_buffer:  # 새 음절 시작: 초성을 기대
            if _is_choseong(current_jamo):
                syllable_buffer.append(current_jamo)
            else:
                # 초성이 아닌 문자가 처음 오면 그대로 결과에 추가
                # (예: 특수문자, 이미 완성된 글자, 또는 단독 모음 등)
                result.append(current_jamo)
            continue

        # syllable_buffer에 초성(L)이 있는 상태: 중성(V)을 기대
        if len(syllable_buffer) == 1:
            l = syllable_buffer[0]
            if _is_jungseong(current_jamo):
                syllable_buffer.append(current_jamo)  # [L, V] 상태가 됨
            else:
                # 중성이 오지 않으면, 이전 초성은 단독으로 처리 (결과에 추가)
                result.append(l)
                syllable_buffer = []  # 버퍼 초기화
                # 현재 자모를 새 음절의 시작으로 다시 처리
                if _is_choseong(current_jamo):
                    syllable_buffer.append(current_jamo)
                else:
                    result.append(current_jamo)
            continue

        # syllable_buffer에 초성(L), 중성(V)이 있는 상태: 종성(T) 또는 새 초성(L')을 기대
        if len(syllable_buffer) == 2:
            l, v = syllable_buffer[0], syllable_buffer[1]

            # 다음 자모가 중성인지 미리 확인
            next_jamo_is_vowel = False
            if (i + 1) < len(jamo_sequence):
                if _is_jungseong(jamo_sequence[i + 1]):
                    next_jamo_is_vowel = True

            # 경우 1: 현재 자모가 종성(T)이 될 수 있고, 다음 자모가 중성이 아님
            # (즉, 현재 자모가 다음 음절의 초성이 될 가능성이 낮음)
            if _is_jongseong(current_jamo) and not next_jamo_is_vowel:
                syllable_buffer.append(current_jamo)  # [L, V, T] 상태
                result.append(_compose_syllable(syllable_buffer[0], syllable_buffer[1], syllable_buffer[2]))
                syllable_buffer = []  # 음절 완성 후 버퍼 초기화
            # 경우 2: 현재 자모가 초성(L')이 될 수 있고, 다음 자모가 중성(V')임
            # (즉, 현재 자모는 새 음절 [L', V']의 시작)
            elif _is_choseong(current_jamo) and next_jamo_is_vowel:
                result.append(_compose_syllable(l, v))  # 이전 음절 [L, V] 완성 (종성 없음)
                syllable_buffer = [current_jamo]  # 현재 자모로 새 음절 시작
            # 경우 3: 현재 자모가 초성(L')이지만 다음 자모가 중성이 아님 (또는 마지막 자모)
            #          또는 현재 자모가 종성(T)으로도 쓰일 수 있음 (경우 1과 유사하나 우선순위)
            elif _is_choseong(current_jamo) and not next_jamo_is_vowel:
                if _is_jongseong(current_jamo):  # 종성으로 사용 가능하다면 종성으로 우선 처리
                    syllable_buffer.append(current_jamo)  # [L, V, T]
                    result.append(_compose_syllable(syllable_buffer[0], syllable_buffer[1], syllable_buffer[2]))
                    syllable_buffer = []
                else:  # 종성으로 사용 불가능한 초성이면 (예: ㄸ, ㅃ, ㅉ)
                    result.append(_compose_syllable(l, v))  # [L,V] 완성
                    syllable_buffer = [current_jamo]  # 새 음절 시작
            # 경우 4: 현재 자모가 중성(V')임
            #          (예: ㄱ ㅏ ㅣ -> '가ㅣ', 이전 음절 [L,V] 완성 후 현재 중성은 단독 처리)
            elif _is_jungseong(current_jamo):
                result.append(_compose_syllable(l, v))  # [L,V] 완성
                syllable_buffer = []  # 버퍼 초기화
                # 현재 중성은 홀로 쓰일 수 없으므로, 'ㅇ'이 생략된 형태로 보거나 그대로 추가
                # 여기서는 일단 그대로 추가 (예: '가'+'ㅣ' -> '가ㅣ')
                # 만약 '이'를 만들고 싶다면 입력이 ['ㅇ','ㅣ'] 여야 함
                result.append(current_jamo)
                # 경우 5: 그 외 (자모가 아닌 문자 등)
            else:
                result.append(_compose_syllable(l, v))  # [L,V] 완성
                syllable_buffer = []
                result.append(current_jamo)  # 현재 문자 그대로 추가
            continue

    # 반복문 종료 후 버퍼에 남은 자모 처리
    if syllable_buffer:
        if len(syllable_buffer) == 1:  # 초성만 남은 경우
            result.append(syllable_buffer[0])
        elif len(syllable_buffer) == 2:  # [L, V]가 남은 경우
            result.append(_compose_syllable(syllable_buffer[0], syllable_buffer[1]))
        elif len(syllable_buffer) == 3:  # [L, V, T]가 남은 경우 (정상적이라면 내부에서 처리되었어야 함)
            result.append(_compose_syllable(syllable_buffer[0], syllable_buffer[1], syllable_buffer[2]))

    return "".join(result)


# --- 예제 사용 ---
if __name__ == '__main__':
    jamo_input1 = ['ㄱ', 'ㅏ', 'ㄴ', 'ㅣ']
    result1 = combine_hangul_jamo(jamo_input1)
    print(f"입력: {jamo_input1} -> 결과: {result1}")  # 예상: 가니

    jamo_input2 = ['ㅎ', 'ㅏ', 'ㄴ', 'ㄱ', 'ㅡ', 'ㄹ']
    result2 = combine_hangul_jamo(jamo_input2)
    print(f"입력: {jamo_input2} -> 결과: {result2}")  # 예상: 한글

    jamo_input3 = ['ㅇ', 'ㅜ', 'ㄹ', 'ㅣ']  # '우리'를 의도하지만, 'ㅜㄹㅣ'가 될 수 있음
    result3 = combine_hangul_jamo(jamo_input3)
    print(f"입력: {jamo_input3} -> 결과: {result3}")  # 예상: 우리 (우 + 리)

    jamo_input4 = ['ㄱ', 'ㅏ', 'ㅂ', 'ㅅ']  # '값'
    result4 = combine_hangul_jamo(jamo_input4)
    print(f"입력: {jamo_input4} -> 결과: {result4}")  # 예상: 값 (JONGSEONG_LIST에 'ㅄ'이 있어야 함)
    # 현재 코드는 '갑ㅅ'이 될 것임. 'ㅄ'을 쓰려면 입력이 ['ㄱ','ㅏ','ㅄ'] 이어야 함.
    # 만약 ['ㅂ','ㅅ'] -> 'ㅄ' 전처리가 필요하다면 별도 함수 추가 필요.
    # 여기서는 JONGSEONG_LIST에 'ㅄ'이 있으므로, 입력이 ['ㄱ','ㅏ','ㅄ']이면 "값"이 됨.
    # ['ㄱ','ㅏ','ㅂ','ㅅ']의 경우 -> 갑ㅅ (현재 로직)

    jamo_input5 = ['ㄱ', 'ㅗ', 'ㅇ', 'ㅏ', 'ㅇ', 'ㅣ']  # '공아이'
    result5 = combine_hangul_jamo(jamo_input5)
    print(f"입력: {jamo_input5} -> 결과: {result5}")  # 예상: 공아이

    jamo_input6 = ['ㄱ', 'ㄱ', 'ㅏ']  # '까'를 의도. CHOSEONG_LIST에 'ㄲ'이 있으므로, 입력이 ['ㄲ','ㅏ']여야 함.
    # 현재 입력 ['ㄱ','ㄱ','ㅏ'] -> 'ㄱ가'
    result6 = combine_hangul_jamo(jamo_input6)
    print(f"입력: {jamo_input6} -> 결과: {result6}")

    # JONGSEONG_LIST에 'ㅄ'이 있고, CHOSEONG_LIST에 'ㄲ'이 있으므로,
    # 이러한 복합 자모는 단일 항목으로 주어져야 합니다.
    jamo_input_complex1 = ['ㄱ', 'ㅏ', 'ㅄ']  # 값
    result_complex1 = combine_hangul_jamo(jamo_input_complex1)
    print(f"입력: {jamo_input_complex1} -> 결과: {result_complex1}")

    jamo_input_complex2 = ['ㄲ', 'ㅏ']  # 까
    result_complex2 = combine_hangul_jamo(jamo_input_complex2)
    print(f"입력: {jamo_input_complex2} -> 결과: {result_complex2}")

    jamo_input_complex3 = ['ㅇ', 'ㅗ', 'ㅏ']  # '와'. JUNGSEONG_LIST에 'ㅘ'가 있으므로, 입력이 ['ㅇ','ㅘ']여야 함.
    # 현재 입력 ['ㅇ','ㅗ','ㅏ'] -> '오ㅏ'
    result_complex3 = combine_hangul_jamo(jamo_input_complex3)
    print(f"입력: {jamo_input_complex3} -> 결과: {result_complex3}")

    jamo_input_complex4 = ['ㅇ', 'ㅘ']  # 와
    result_complex4 = combine_hangul_jamo(jamo_input_complex4)
    print(f"입력: {jamo_input_complex4} -> 결과: {result_complex4}")
