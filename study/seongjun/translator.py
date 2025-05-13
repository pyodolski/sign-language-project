from googletrans import Translator
# 구글 번역 라이브러리에서 Translator 클래스 불러오기
# (텍스트를 다양한 언어로 번역할 수 있는 객체)


translator = Translator()
# Translator 객체 생성: 이후 translate 메서드를 사용할 수 있게 됨

def translate_to_korean(text):
    result = translator.translate(text, src='en', dest='ko')
    # translate(): 구글 번역 API 호출
    # src='en': 입력 언어는 영어
    # dest='ko': 출력 언어는 한국어
    # result는 번역된 내용과 관련된 정보가 담긴 객체
    return result.text # 번역된 한국어 문장(result.text)만 반환
