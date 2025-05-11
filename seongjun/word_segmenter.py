from wordsegment import load, segment
# wordsegment 라이브러리에서 load, segment 함수 불러오기
# load(): 단어 사전 데이터 로드
# segment(): 공백 없이 이어진 알파벳을 의미 있는 영어 단어로 분리해주는 함수

load() # 사전 단어 데이터를 메모리에 불러옴 (segment를 사용하기 전에 반드시 한 번 실행해야 함)

def build_english_sentence(alphabet_stream):
    # 공백 없는 알파벳 문자열을 받아서 의미 있는 영어 단어들로 나누는 함수
    # alphabet_stream: 예측된 알파벳들이 이어진 문자열 (예: 'helloworld')
    
    return " ".join(segment(alphabet_stream))
    # segment(): 알파벳 스트림을 의미 있는 단어 리스트로 나눔
    # " ".join(): 나눠진 단어들을 공백으로 연결하여 하나의 문장으로 만듦


# 이 코드는 알파벳 예측 결과들을 이어 붙인 문자열을 받아,
# wordsegment 라이브러리를 이용해 의미 있는 영어 단어들로 분리해주는 기능을 한다.
# 예: 'helloworld' → 'hello world'
# 이를 통해 손으로 알파벳을 하나씩 입력해도 자연스러운 문장으로 만들 수 있다.
