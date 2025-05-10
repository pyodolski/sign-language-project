from wordsegment import load, segment

load()

def build_english_sentence(alphabet_stream):
    return " ".join(segment(alphabet_stream))
