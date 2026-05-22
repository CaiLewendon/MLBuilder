"""Downgrade CONV_2D op-code version flag to 3 so edgetpu_compiler 16.0 accepts the model.
Run after surgery_grouped_to_depthwise.py."""
import sys, flatbuffers
from tensorflow.lite.python import schema_py_generated as schema

IN  = sys.argv[1] if len(sys.argv) > 1 else 'export/FullDataSetProd_saved_model/FullDataSetProd_full_integer_quant_dwfix.tflite'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'export/FullDataSetProd_saved_model/FullDataSetProd_full_integer_quant_dwfix_v3.tflite'

buf = open(IN, 'rb').read()
m = schema.Model.GetRootAs(buf, 0)
mT = schema.ModelT.InitFromObj(m)

downgraded = 0
for oc in mT.operatorCodes:
    if oc.builtinCode == 3 and oc.version > 3:  # CONV_2D
        print(f"downgrading CONV_2D op-code version {oc.version} -> 3")
        oc.version = 3
        downgraded += 1

print(f"downgraded {downgraded} op-code entry(ies)")

b = flatbuffers.Builder(1024)
b.Finish(mT.Pack(b), b"TFL3")
open(OUT, 'wb').write(bytes(b.Output()))
print(f"wrote {OUT}")
