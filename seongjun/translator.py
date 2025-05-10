from googletrans import Translator

translator = Translator()

def translate_to_korean(text):
    result = translator.translate(text, src='en', dest='ko')
    return result.text
