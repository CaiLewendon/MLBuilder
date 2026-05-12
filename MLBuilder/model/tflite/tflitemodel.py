from typing import Literal, Optional, Union
import numpy as np
import cv2
import os
from MLBuilder.model.mlmodel import MLModel, system, allow, disallow

try:
    from ai_edge_litert.interpreter import Interpreter

    INTERPRETER_EXSITS = True
except ImportError as e:
    INTERPRETER_EXSITS = False


class TFLiteModel(MLModel):
    def __init__(self, path: str):
        super().__init__(path)
        self._normalize = False
        self._started = False

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
        from ultralytics import YOLO  # pyright: ignore[reportPrivateImportUsage]

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
        except:
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
                format="edgetpu", imgsz=imgsz, project=os.path.join(archive_dir, "out")
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
    def allocate(self, tpu=False):
        if not INTERPRETER_EXSITS:
            raise RuntimeError("INTERPRETER_EXSITS FLASE")
        self._intepreter = Interpreter(model_path=self.path)
        self._intepreter.allocate_tensors()

        self._input_details = self._intepreter.get_input_details()
        self._output_detail = self._intepreter.get_output_details()
        self._input_dtype = self._input_details[0]["dtype"]
        self._input_quant = self._input_details[0].get("quantization", (0.0, 0))
        self._output_quant = self._output_detail[0].get("quantization", (0.0, 0))

        if self._input_dtype in (np.float32, np.float16):
            self._normalize = True

        self._started = True

    @system("linux")
    def detect(self, data: np.ndarray, nms=False, tol=0.25):
        if not INTERPRETER_EXSITS:
            raise RuntimeError("INTERPRETER_EXSITS FLASE")
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

        if self._input_dtype in (np.float32, np.float16):
            model_input = (padded_image.astype(np.float32) / 255.0).astype(self._input_dtype)
        elif self._input_dtype in (np.int8, np.uint8, np.int16, np.uint16):
            q_scale, q_zero = self._input_quant
            if q_scale is None or float(q_scale) <= 0.0:
                raise RuntimeError(
                    f"Invalid quantized input scale for dtype={self._input_dtype}: {q_scale}"
                )
            # Override for field debugging:
            #   MLBUILDER_TFLITE_INPUT_REAL_SPACE=normalized|pixels|auto
            mode = os.getenv("MLBUILDER_TFLITE_INPUT_REAL_SPACE", "auto").strip().lower()
            if mode == "normalized":
                real_input = padded_image.astype(np.float32) / 255.0
            elif mode == "pixels":
                real_input = padded_image.astype(np.float32)
            else:
                # Heuristic: small scales typically imply normalized [0,1] real-space input.
                if float(q_scale) < 0.02:
                    real_input = padded_image.astype(np.float32) / 255.0
                else:
                    real_input = padded_image.astype(np.float32)
            q = np.round(real_input / float(q_scale) + float(q_zero))
            limits = np.iinfo(self._input_dtype)
            q = np.clip(q, limits.min, limits.max)
            model_input = q.astype(self._input_dtype)
        else:
            model_input = padded_image.astype(self._input_dtype)

        model_input = np.expand_dims(model_input, axis=0)

        self._intepreter.set_tensor(
            self._input_details[0]["index"],
            model_input,
        )
        self._intepreter.invoke()
        raw_out = self._intepreter.get_tensor(self._output_detail[0]["index"])
        out_dtype = self._output_detail[0]["dtype"]
        if out_dtype in (np.int8, np.uint8, np.int16, np.uint16):
            out_scale, out_zero = self._output_quant
            if out_scale is not None and float(out_scale) > 0.0:
                raw_out = (raw_out.astype(np.float32) - float(out_zero)) * float(out_scale)

        if not nms:
            detections = raw_out[0]
            if detections.ndim != 2 or detections.shape[1] < 6:
                return []
            valid = detections[detections[:, 4] > tol]

            output = []

            for detection in valid:
                x_min = int((detection[0] * input_w - pad_left) / scale)
                y_min = int((detection[1] * input_h - pad_top) / scale)
                x_max = int((detection[2] * input_w - pad_left) / scale)
                y_max = int((detection[3] * input_h - pad_top) / scale)

                confidence = detection[4]
                class_id = detection[5]

                output.append(
                    {
                        "id": int(class_id),
                        "confidence": float(confidence),
                        "bbox": ((x_min, y_min), (x_max, y_max)),
                    }
                )

            return output
        else:
            out = raw_out[0]
            if out.ndim != 2:
                return []

            if out.shape[0] > out.shape[1]:
                predictions = out
            else:
                predictions = out.T

            if predictions.shape[1] < 5:
                return []

            boxes = predictions[:, :4]
            class_scores = predictions[:, 4:]
            if class_scores.shape[1] == 0:
                return []

            class_ids = np.argmax(class_scores, axis=1)
            confidences = np.max(class_scores, axis=1)

            mask = confidences > tol
            filtered_boxes = boxes[mask]
            filtered_confidences = confidences[mask]
            filtered_class_ids = class_ids[mask]

            if len(filtered_boxes) == 0:
                return []

            x_center = filtered_boxes[:, 0]
            y_center = filtered_boxes[:, 1]
            w = filtered_boxes[:, 2]
            h = filtered_boxes[:, 3]

            # Some TFLite exports emit normalized xywh [0..1], others emit pixel-space xywh.
            # Normalize handling here so downstream bbox remap is always in padded-input pixel space.
            if np.max(np.abs(filtered_boxes)) <= 2.0:
                x_center = x_center * input_w
                y_center = y_center * input_h
                w = w * input_w
                h = h * input_h

            x_min = x_center - w / 2
            y_min = y_center - h / 2
            x_max = x_center + w / 2
            y_max = y_center + h / 2

            xyxy_boxes = np.stack([x_min, y_min, x_max, y_max], axis=1)
            xywh_boxes = np.stack(
                [x_min, y_min, np.maximum(0.0, w), np.maximum(0.0, h)], axis=1
            )

            indices = cv2.dnn.NMSBoxes(
                xywh_boxes.tolist(),
                filtered_confidences.tolist(),
                tol,
                0.45,
            )

            if len(indices) == 0:
                return []

            if isinstance(indices, np.ndarray):
                indices = indices.flatten()
            else:
                indices = np.array(indices).flatten()

            final_boxes = xyxy_boxes[indices]
            final_confidences = filtered_confidences[indices]
            final_class_ids = filtered_class_ids[indices]

            output = []
            for box, confidence, class_id in zip(
                final_boxes, final_confidences, final_class_ids
            ):
                x_min_scaled = int((box[0] - pad_left) / scale)
                y_min_scaled = int((box[1] - pad_top) / scale)
                x_max_scaled = int((box[2] - pad_left) / scale)
                y_max_scaled = int((box[3] - pad_top) / scale)

                output.append(
                    {
                        "id": int(class_id),
                        "confidence": float(confidence),
                        "bbox": (
                            (x_min_scaled, y_min_scaled),
                            (x_max_scaled, y_max_scaled),
                        ),
                    }
                )

            return output
