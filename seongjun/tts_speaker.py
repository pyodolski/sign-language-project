from gtts import gTTS
# gTTS(Google Text-to-Speech) 라이브러리에서 gTTS 클래스 불러오기
# (텍스트를 음성(mp3) 파일로 변환해주는 기능 제공)

import os # os 모듈 불러오기: 시스템 명령어 실행 (음성 파일 재생용으로 사용)

def speak_korean(text): #text: 음성으로 읽을 문장
    tts = gTTS(text=text, lang='ko') # gTTS 객체 생성: 입력된 text를 한국어(lang='ko')로 설정하여 음성으로 변환
    tts.save("speech.mp3") # 변환된 음성을 mp3 파일로 저장 ("speech.mp3"라는 이름으로 저장됨
    os.system("omxplayer speech.mp3")  
    # 저장된 음성 파일을 시스템 명령으로 재생 (omxplayer는 라즈베리파이 전용 음성 플레이어)
    # ※ 다른 환경에서는 mpg321, playsound 등으로 교체 가능



# 이 코드는 입력된 한국어 문장을 gTTS를 사용해 음성(mp3) 파일로 변환하고,
# 라즈베리파이에서 omxplayer를 이용해 그 음성을 재생하는 기능을 수행한다.
# 한국어 텍스트를 자동으로 말해주는 TTS(Text-to-Speech) 기능이다.
