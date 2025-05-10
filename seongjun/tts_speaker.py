from gtts import gTTS
import os

def speak_korean(text):
    tts = gTTS(text=text, lang='ko')
    tts.save("speech.mp3")
    os.system("omxplayer speech.mp3")  # 또는 "mpg321 speech.mp3"
