import time

buffer = ""
prev_char = ""
last_time = time.time()

def add_char(char):
    global buffer, prev_char, last_time
    if char == prev_char and time.time() - last_time < 1.0:
        return None
    prev_char = char
    last_time = time.time()
    buffer += char
    return buffer
