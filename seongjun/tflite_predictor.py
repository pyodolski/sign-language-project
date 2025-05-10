import numpy as np
import tflite_runtime.interpreter as tflite

class ASLPredictor:
    def __init__(self, model_path="asl_model.tflite"):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, sequence):
        input_data = np.array(sequence, dtype=np.float32).reshape(1, 30, 63)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        return chr(np.argmax(output) + ord('A'))
