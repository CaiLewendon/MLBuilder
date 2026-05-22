"""Convert any CONV_2D op with filter shape (C,kH,kW,1) and in_channels=C into
DEPTHWISE_CONV_2D so edgetpu_compiler 16.0 accepts the model."""
import sys, numpy as np, flatbuffers
from tensorflow.lite.python import schema_py_generated as schema

IN  = sys.argv[1] if len(sys.argv) > 1 else 'export/FullDataSetProd_saved_model/FullDataSetProd_full_integer_quant_tf215.tflite'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'export/FullDataSetProd_saved_model/FullDataSetProd_full_integer_quant_dwfix.tflite'

buf = open(IN, 'rb').read()
m = schema.Model.GetRootAs(buf, 0)
mT = schema.ModelT.InitFromObj(m)

conv2d_idx = next((i for i, oc in enumerate(mT.operatorCodes) if oc.builtinCode == 3), None)
dwconv_idx = next((i for i, oc in enumerate(mT.operatorCodes) if oc.builtinCode == 4), None)
assert conv2d_idx is not None, "no CONV_2D op_code"
if dwconv_idx is None:
    new_oc = schema.OperatorCodeT()
    new_oc.builtinCode = 4
    new_oc.deprecatedBuiltinCode = 4
    new_oc.version = 3
    mT.operatorCodes.append(new_oc)
    dwconv_idx = len(mT.operatorCodes) - 1
    print(f"added DEPTHWISE_CONV_2D op_code at idx {dwconv_idx}")
print(f"CONV_2D idx={conv2d_idx}  DEPTHWISE_CONV_2D idx={dwconv_idx}")

converted = 0
for sg_i, sg in enumerate(mT.subgraphs):
    for op_i, op in enumerate(sg.operators):
        if op.opcodeIndex != conv2d_idx:
            continue
        ft = sg.tensors[op.inputs[1]]
        in_t = sg.tensors[op.inputs[0]]
        if len(ft.shape) != 4 or len(in_t.shape) != 4:
            continue
        C_out, kH, kW, per_g = list(ft.shape)
        in_c = in_t.shape[3]
        if per_g != 1 or in_c != C_out:
            continue   # not a depthwise candidate

        print(f"  subgraph={sg_i} op={op_i}  filter=({C_out},{kH},{kW},1) input_C={in_c}  -> DEPTHWISE")

        # 1. Reshape filter weights: (C_out, kH, kW, 1) -> (1, kH, kW, C_out)
        buffer = mT.buffers[ft.buffer]
        raw = bytes(buffer.data)
        arr = np.frombuffer(raw, dtype=np.int8).reshape(C_out, kH, kW, 1)
        new_arr = np.transpose(arr, (3, 1, 2, 0)).copy()  # (1, kH, kW, C_out)
        buffer.data = np.frombuffer(new_arr.tobytes(), dtype=np.uint8).tolist()
        ft.shape = [1, kH, kW, C_out]
        if ft.quantization is not None:
            ft.quantization.quantizedDimension = 3

        # 2. Build DepthwiseConv2DOptions from the existing Conv2DOptions
        old = op.builtinOptions
        new_opts = schema.DepthwiseConv2DOptionsT()
        new_opts.padding = old.padding
        new_opts.strideW = old.strideW
        new_opts.strideH = old.strideH
        new_opts.depthMultiplier = 1
        new_opts.fusedActivationFunction = old.fusedActivationFunction
        new_opts.dilationWFactor = old.dilationWFactor
        new_opts.dilationHFactor = old.dilationHFactor
        op.builtinOptions = new_opts
        op.builtinOptionsType = schema.BuiltinOptions.DepthwiseConv2DOptions
        op.opcodeIndex = dwconv_idx
        converted += 1

print(f"converted {converted} op(s)")

b = flatbuffers.Builder(1024)
b.Finish(mT.Pack(b), b"TFL3")
open(OUT, 'wb').write(bytes(b.Output()))
print(f"wrote {OUT}")
