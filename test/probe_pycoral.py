from pycoral.utils.edgetpu import make_interpreter

MODEL = "/home/pi/model_edgetpu.tflite"

print("Creating interpreter...")
interpreter = make_interpreter(MODEL)
print("Interpreter created")

print("Allocating tensors...")
interpreter.allocate_tensors()

print("SUCCESS: TPU interpreter initialized")
print("Input details:", interpreter.get_input_details())
print("Output details:", interpreter.get_output_details())
