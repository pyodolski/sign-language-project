import time  # 현재 시간을 측정하기 위해 time 모듈을 가져옴

buffer = ""  # 누적된 알파벳들을 저장하는 전역 문자열
prev_char = ""  # 이전에 입력된 문자 저장용 변수
last_time = time.time()  # 마지막으로 문자가 입력된 시간 초기화

def add_char(char):  # 새로운 문자를 입력받아 조건에 따라 누적하는 함수
    global buffer, prev_char, last_time  # 함수 안에서 전역 변수들을 수정하기 위해 선언
    if char == prev_char and time.time() - last_time < 1.0:  # 같은 문자가 1초 이내에 반복되면 무시
        return None  # 중복 입력이므로 아무것도 반환하지 않음
    prev_char = char  # 이전 문자 변수에 현재 문자를 저장
    last_time = time.time()  # 마지막 입력 시간 갱신
    buffer += char  # 새로운 문자를 buffer에 추가
    return buffer  # 현재까지 누적된 문자열을 반환


# 동작방식
# 이 코드는 예측된 알파벳 문자를 받아서,
# 같은 문자가 너무 빠르게 반복되면 무시하고,
# 나머지는 문자열로 차곡차곡 누적해서 반환하는 역할을 한다
