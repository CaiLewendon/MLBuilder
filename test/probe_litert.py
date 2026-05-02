from ai_edge_litert.interpreter import Interpreter, load_delegate

MODEL = "/home/pi/model_edgetpu.tflite"

print("Loading delegate...")
delegate = load_delegate("libedgetpu.so.1")
print("Delegate loaded OK")

print("Creating interpreter...")
interpreter = Interpreter(
    model_path=MODEL,
    experimental_delegates=[delegate],
)
print("Interpreter created")

print("Allocating tensors...")
interpreter.allocate_tensors()

print("SUCCESS: TPU interpreter initialized")
print("Input details:", interpreter.get_input_details())
print("Output details:", interpreter.get_output_details())
