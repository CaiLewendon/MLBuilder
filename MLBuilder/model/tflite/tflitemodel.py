from typing import Literal, Optional
import numpy as np
import cv2
from MLBuilder.model.mlmodel import MLModel, system, allow, disallow

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate

    INTERPRETER_EXSITS = True
except ImportError:
    try:
        from ai_edge_litert.interpreter import Interpreter, load_delegate

        INTERPRETER_EXSITS = True
    except ImportError:
        INTERPRETER_EXSITS = False

print(f"[TFLITEMODEL] Loaded from: {__file__}")


class TFLiteModel(MLModel):
    def __init__(self, path: str):
        super().__init__(path)
        self._normalize = False
        self._started = False
        self._using_tpu = False
        self._delegate = None
        self._input_quant_scale = 0.0
        self._input_quant_zero_point = 0
        self._output_quant_scale = 0.0
        self._output_quant_zero_point = 0

    @allow("pt")
    @system("linux")
    def build(
        self,
        outdir: str,
        imgsz: int,
        quant: Optional[Literal["fp32", "fp16", "int8"]],
        data: str = "coco8.yaml",
        edge: bool = False,
    ) -> str:
        if not INTERPRETER_EXSITS:
            raise RuntimeError("INTERPRETER_EXSITS FLASE")

        import os
        import shutil
        from ultralytics import YOLO

        if not os.path.isdir(outdir):
            raise ValueError(f"'{outdir}' does not exsist")
        if not os.access(outdir, os.W_OK):
            raise RuntimeError(f"'{outdir} is not writable'")

        outdir = os.path.abspath(outdir)
        data = os.path.abspath(data) if data != "coco8.yaml" else data
        model_path = os.path.abspath(self.path)

        working_dir = os.getcwd()
        archive_dir = os.path.join(outdir, "build")
        if not os.path.isdir(archive_dir):
            os.mkdir(archive_dir)
        os.chdir(archive_dir)

        try:
            model = YOLO(model_path)
        except Exception:
            model = YOLO(self.path)

        if not edge:
            model_out = model.export(
                format="tflite",
                imgsz=imgsz,
                half=True if quant == "fp16" else False,
                int8=True if quant == "int8" else False,
                nms=True,
                data=data,
                project=os.path.join(archive_dir, "out"),
            )
        else:
            model_out = model.export(
                format="edgetpu",
                imgsz=imgsz,
                project=os.path.join(archive_dir, "out"),
            )

        model_dir = os.path.join(archive_dir, model_out)
        os.chdir(working_dir)

        export_dir = os.path.join(outdir, "export")
        if not os.path.isdir(export_dir):
            os.mkdir(export_dir)
        shutil.copy(model_dir, export_dir)

        model_out = os.path.join(export_dir, os.path.basename(model_dir))
        return model_out

    @disallow("pt")
    @system("linux")
    def allocate(self, tpu: bool = False):
        if not INTERPRETER_EXSITS:
            raise RuntimeError("INTERPRETER_EXSITS FALSE")

        self._normalize = False
        self._using_tpu = False
        self._delegate = None

        last_error = None

        if tpu:
            try:
                self._delegate = load_delegate("libedgetpu.so.1")
                self._intepreter = Interpreter(
                    model_path=self.path,
                    experimental_delegates=[self._delegate],
                )
                self._using_tpu = True
                print(f"[ALLOCATE] Using Edge TPU for model: {self.path}")
            except Exception as e:
                last_error = e
                self._using_tpu = False
                self._delegate = None
                print(f"[ALLOCATE] Edge TPU unavailable, falling back to CPU: {e}")

        if not self._using_tpu:
            self._intepreter = Interpreter(model_path=self.path)
            print(f"[ALLOCATE] Using CPU for model: {self.path}")

        self._intepreter.allocate_tensors()

        self._input_details = self._intepreter.get_input_details()
        self._output_detail = self._intepreter.get_output_details()

        input_quant = self._input_details[0].get("quantization", (0.0, 0))
        output_quant = self._output_detail[0].get("quantization", (0.0, 0))
        self._input_quant_scale, self._input_quant_zero_point = input_quant
        self._output_quant_scale, self._output_quant_zero_point = output_quant

        if self._input_details[0]["dtype"] in (np.float32, np.float16):
            self._normalize = True

        self._started = True

        print(f"[ALLOCATE] TPU active: {self._using_tpu}")
        print(f"[ALLOCATE] Input dtype: {self._input_details[0]['dtype']}")
        print(f"[ALLOCATE] Input shape: {self._input_details[0]['shape']}")
        if last_error is not None:
            print(f"[ALLOCATE] TPU init error was: {last_error}")

    def _map_box_back(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        input_w: int,
        input_h: int,
        pad_left: int,
        pad_top: int,
        scale: float,
        data_w: int,
        data_h: int,
    ) -> tuple[int, int, int, int]:
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 2.0:
            x1 *= input_w
            x2 *= input_w
            y1 *= input_h
            y2 *= input_h

        x1 = int((x1 - pad_left) / scale)
        y1 = int((y1 - pad_top) / scale)
        x2 = int((x2 - pad_left) / scale)
        y2 = int((y2 - pad_top) / scale)

        x1 = max(0, min(x1, data_w - 1))
        y1 = max(0, min(y1, data_h - 1))
        x2 = max(0, min(x2, data_w - 1))
        y2 = max(0, min(y2, data_h - 1))
        return x1, y1, x2, y2

    @system("linux")
    def detect(self, data: np.ndarray, nms=False, tol=0.25):
        if not INTERPRETER_EXSITS:
            raise RuntimeError("INTERPRETER_EXSITS FLASE")
        if not self._started:
            raise RuntimeError("Model has not been allocated")

        img = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

        input_h, input_w = self._input_details[0]["shape"][1:3]
        data_h, data_w = data.shape[0:2]

        scale_h = input_h / data_h
        scale_w = input_w / data_w
        scale = min(scale_h, scale_w)

        new_h = int(data_h * scale)
        new_w = int(data_w * scale)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pad_h = input_h - new_h
        pad_w = input_w - new_w
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        padded_image = cv2.copyMakeBorder(
            resized_img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

        input_dtype = self._input_details[0]["dtype"]
        if input_dtype in (np.float32, np.float16):
            model_input = (padded_image.astype(np.float32) / 255.0).astype(input_dtype)
        else:
            model_input = padded_image.astype(np.float32) / 255.0
            if self._input_quant_scale > 0:
                model_input = (
                    model_input / self._input_quant_scale + self._input_quant_zero_point
                )
            info = np.iinfo(input_dtype)
            model_input = np.clip(np.round(model_input), info.min, info.max).astype(
                input_dtype
            )

        model_input = np.expand_dims(model_input, axis=0)

        self._intepreter.set_tensor(self._input_details[0]["index"], model_input)
        self._intepreter.invoke()
        raw_out = self._intepreter.get_tensor(self._output_detail[0]["index"])

        out_dtype = self._output_detail[0]["dtype"]
        if out_dtype in (np.int8, np.uint8) and self._output_quant_scale > 0:
            raw_out = (
                raw_out.astype(np.float32) - self._output_quant_zero_point
            ) * self._output_quant_scale

        out = raw_out[0]

        print(f"[OUT] shape={raw_out.shape} max={float(raw_out.max()):.4f} mean={float(raw_out.mean()):.4f}")

        # Postprocessed format: [N,6], but layout can vary by export/runtime:
        # [x1,y1,x2,y2,conf,cls] or [x1,y1,x2,y2,cls,conf]
        # [cx,cy,w,h,conf,cls]   or [cx,cy,w,h,cls,conf]
        if out.ndim == 2 and 6 <= out.shape[1] <= 16 and out.shape[0] > out.shape[1]:
            layouts = [
                ("xyxy", 4, 5),
                ("xyxy", 5, 4),
                ("xywh", 4, 5),
                ("xywh", 5, 4),
            ]

            best_results = []
            best_count = -1

            for box_fmt, conf_idx, cls_idx in layouts:
                if conf_idx >= out.shape[1] or cls_idx >= out.shape[1]:
                    continue

                conf_col = out[:, conf_idx]
                valid = out[conf_col > tol]
                results = []

                for det in valid:
                    if box_fmt == "xyxy":
                        x1f, y1f, x2f, y2f = (
                            float(det[0]),
                            float(det[1]),
                            float(det[2]),
                            float(det[3]),
                        )
                    else:
                        cx, cy, bw, bh = (
                            float(det[0]),
                            float(det[1]),
                            float(det[2]),
                            float(det[3]),
                        )
                        x1f = cx - bw / 2.0
                        y1f = cy - bh / 2.0
                        x2f = cx + bw / 2.0
                        y2f = cy + bh / 2.0

                    x1, y1, x2, y2 = self._map_box_back(
                        x1f,
                        y1f,
                        x2f,
                        y2f,
                        input_w,
                        input_h,
                        pad_left,
                        pad_top,
                        scale,
                        data_w,
                        data_h,
                    )
                    if x2 <= x1 or y2 <= y1:
                        continue

                    class_id = int(det[cls_idx])
                    confidence = float(det[conf_idx])

                    results.append(
                        {
                            "id": class_id,
                            "confidence": confidence,
                            "bbox": ((x1, y1), (x2, y2)),
                        }
                    )

                if len(results) > best_count:
                    best_count = len(results)
                    best_results = results

            return best_results

        # Raw head format: [5,8400] or [8400,5], needs runtime NMS
        predictions = out if out.shape[0] > out.shape[1] else out.T
        if predictions.ndim != 2 or predictions.shape[1] < 5:
            return []

        boxes_xywh = predictions[:, :4].astype(np.float32)
        class_scores = predictions[:, 4:]

        if class_scores.shape[1] == 1:
            class_ids = np.zeros(class_scores.shape[0], dtype=np.int32)
            confidences = class_scores[:, 0]
        else:
            class_ids = np.argmax(class_scores, axis=1)
            confidences = np.max(class_scores, axis=1)

        mask = confidences > tol
        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes_xywh) == 0:
            return []

        if boxes_xywh[:, :4].max() <= 2.0:
            boxes_xywh[:, [0, 2]] *= input_w
            boxes_xywh[:, [1, 3]] *= input_h

        x = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
        y = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
        w = boxes_xywh[:, 2]
        h = boxes_xywh[:, 3]
        nms_boxes = np.stack([x, y, w, h], axis=1)

        indices = cv2.dnn.NMSBoxes(
            nms_boxes.tolist(),
            confidences.tolist(),
            float(tol),
            0.45,
        )
        if len(indices) == 0:
            return []

        indices = np.array(indices).flatten()
        results = []
        for i in indices:
            bx, by, bw, bh = nms_boxes[i]
            x1, y1, x2, y2 = self._map_box_back(
                float(bx),
                float(by),
                float(bx + bw),
                float(by + bh),
                input_w,
                input_h,
                pad_left,
                pad_top,
                scale,
                data_w,
                data_h,
            )
            if x2 <= x1 or y2 <= y1:
                continue
            results.append(
                {
                    "id": int(class_ids[i]),
                    "confidence": float(confidences[i]),
                    "bbox": ((x1, y1), (x2, y2)),
                }
            )

        return results

    def using_tpu(self) -> bool:
        return self._using_tpu
