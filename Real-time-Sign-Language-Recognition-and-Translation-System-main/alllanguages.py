import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from collections import Counter
import time
from googletrans import Translator
from spellchecker import SpellChecker

model_dict = pickle.load(open(r'C:\Users\pavan\OneDrive\Desktop\signlanguage\SIGNLANGUAGE recognition AND TELUGU CONVERSION\model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to open the camera.")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Thank You',
    27: 'Nice To Meet You'
}

languages = {
    'English': 'en',
    'French': 'fr',
    'Spanish': 'es',
    'German': 'de',
    'Hindi': 'hi',
    'Telugu': 'te',  # Example of Indian language
    'Tamil': 'ta',   # Example of Indian language
    'Kannada': 'kn', # Example of Indian language
    'Malayalam': 'ml' # Example of Indian language
}

root = tk.Tk()
root.title("Sign Language to Text Conversion")

frame_delay = 100  # Delay between frames in milliseconds

frame_layout = tk.Frame(root, padx=20, pady=20)
frame_layout.pack()

frame_english = tk.Frame(frame_layout, padx=20, pady=20, bg="lightgray")
frame_english.pack(side=tk.LEFT, padx=10, pady=10)

english_heading_label = tk.Label(frame_english, text="English Conversion", font=("Helvetica", 18), bg="lightgray")
english_heading_label.pack()

recognized_word_label = tk.Label(frame_english, text="", font=("Helvetica", 24), fg="black", bg="lightgray")
recognized_word_label.pack()

frame_languages = tk.Frame(frame_layout, padx=20, pady=20, bg="lightgray")
frame_languages.pack(side=tk.LEFT, padx=10, pady=10)

language_label = tk.Label(frame_languages, text="Select Language:", font=("Helvetica", 14), bg="lightgray")
language_label.pack()

selected_language = tk.StringVar(root)
selected_language.set('English')  # Default selection

language_options = tk.OptionMenu(frame_languages, selected_language, *languages.keys())
language_options.config(font=("Helvetica", 12), bg="lightgray")
language_options.pack()

frame_translation = tk.Frame(frame_layout, padx=20, pady=20, bg="lightgray")
frame_translation.pack(side=tk.LEFT, padx=10, pady=10)

translation_heading_label = tk.Label(frame_translation, text="Translation", font=("Helvetica", 18), bg="lightgray")
translation_heading_label.pack()

translation_label = tk.Label(frame_translation, text="", font=("Helvetica", 24), fg="black", bg="lightgray")
translation_label.pack()

recognized_word = ""
last_symbol_time = time.time()
translation_shown = False
camera_visible = True  # Flag to track if the camera feed window is visible

translator = Translator()

spell = SpellChecker()

letter_buffer = []

def correct_spelling(sentence):
    corrected_sentence = ""
    words = sentence.split()
    for word in words:
        corrected_word = spell.correction(word)
        corrected_sentence += corrected_word + " "
    return corrected_sentence.strip()

def hide_camera_feed():
    global camera_visible
    cv2.destroyAllWindows()
    camera_visible = False

def show_camera_feed():
    global camera_visible
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    camera_visible = True

def show_translation(sentence, dest_language_code):
    global translation_shown
    if not translation_shown:
        translation_shown = True

        corrected_sentence = correct_spelling(sentence)

        try:
            translation_result = translator.translate(corrected_sentence, src='en', dest=dest_language_code)
            if translation_result and translation_result.text:
                translated_text = translation_result.text
                translation_label.config(text=translated_text, fg="black")
            else:
                print("Translation result or text is None.")
        except Exception as e:
            print(f"Translation error: {e}")

        root.after(6000, clear_labels)
        root.after(6000, show_camera_feed)

def clear_labels():
    global recognized_word, translation_shown

    recognized_word_label.config(text="", fg="black")
    translation_label.config(text="", fg="black")

    recognized_word = ""
    translation_shown = False

def update_recognized_word():
    global recognized_word, letter_buffer, last_symbol_time

    if letter_buffer:
        most_common_letter = Counter(letter_buffer).most_common(1)[0][0]
        recognized_word += most_common_letter
        recognized_word_label.config(text=recognized_word, fg="black")

    letter_buffer = []

    if time.time() - last_symbol_time > 4:
        recognized_word_label.config(fg="red")
        recognized_word += " "
        recognized_word_label.config(text=recognized_word, fg="red")

    root.after(5000, update_recognized_word)

    if time.time() - last_symbol_time > 7:
        recognized_word_label.config(fg="blue")

        # Hide the camera feed during translation
        hide_camera_feed()

        selected_lang = selected_language.get()
        if selected_lang in languages:
            show_translation(recognized_word, languages[selected_lang])

root.after(frame_delay, update_recognized_word)

while True:
    try:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()

        if not ret:
            raise Exception("Error: Unable to read frame from the camera.")

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            last_symbol_time = time.time()

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])

            predicted_character = labels_dict[int(prediction[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

            letter_buffer.append(predicted_character)

        if camera_visible:
            cv2.imshow('frame', frame)

        cv2.waitKey(1)

    except Exception as e:
        print(e)

    print("Recognized Word:", recognized_word)

    root.update_idletasks()
    root.update()

cap.release()
root.destroy()
