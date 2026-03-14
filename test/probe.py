import tflite_runtime.interpreter as tflite

delegate = tflite.load_delegate("libedgetpu.so.1", {"device": "usb:0"})
interpreter = tflite.Interpreter(
    model_path="/home/pi/model_edgetpu.tflite",
    experimental_delegates=[delegate],
)
interpreter.allocate_tensors()
print("TPU interpreter initialized")