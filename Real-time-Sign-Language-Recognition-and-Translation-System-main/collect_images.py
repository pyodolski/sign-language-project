import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 28  # Increased to accommodate 'Thank You' and 'Nice To Meet You'
dataset_size = 100

# Alphabet and additional messages
characters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
              'Thank You', 'Nice To Meet You']

# Attempt to open the camera
cap = cv2.VideoCapture(0)  # Use the default camera (index 0)

# Check if the camera was opened successfully
if not cap.isOpened():
    print("Error: Unable to open the camera.")
    exit()

for j, current_character in enumerate(characters):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {} ({})'.format(j, current_character))

    print('Press "S" when ready.')
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        message = 'Ready? Press "S"! To skip, press "N" :)'
        if current_character == 'Thank You':
            message = 'Thank You!'
        elif current_character == 'Nice To Meet You':
            message = 'Nice To Meet You!'

        cv2.putText(frame, message, (50, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 155, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        # Exit if 'S' is pressed
        if cv2.waitKey(25) == ord('s'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            break

        # Display the current character on the frame
        cv2.putText(frame, 'Character: {}'.format(current_character), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                    (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        # Skip taking pictures if 'N' is pressed
        key = cv2.waitKey(25)
        if key == ord('n'):
            print('Skipping pictures for class {} ({})'.format(j, current_character))
            break

        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
        counter += 1

# Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()
