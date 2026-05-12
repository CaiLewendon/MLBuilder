import tflite_runtime.interpreter as tfl

m = tfl.Interpreter('../target_detector_int8_edgetpu.tflite',
                    experimental_delegates=[tfl.load_delegate('libedgetpu.so.1')])
m.allocate_tensors()
i = m.get_input_details()[0]
o = m.get_output_details()[0]
print('input :', i['shape'], i['dtype'], 'quant=', i['quantization'])
print('output:', o['shape'], o['dtype'], 'quant=', o['quantization'])