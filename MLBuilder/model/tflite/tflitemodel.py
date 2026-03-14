from typing import Literal, Optional
import numpy as np
import cv2
from MLBuilder.model.mlmodel import MLModel, system, allow, disallow

try:
    from ai_edge_litert.interpreter import Interpreter, load_delegate
    INTERPRETER_EXSITS = True
except ImportError:
    INTERPRETER_EXSITS = False


class TFLiteModel(MLModel):
    def __init__(self, path: str):
        super().__init__(path)
        self._normalize = False
        self._started = False
        self._using_tpu = False
        self._delegate = None

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

        if self._input_details[0]["dtype"] in (np.float32, np.float16):
            self._normalize = True

        self._started = True

        print(f"[ALLOCATE] TPU active: {self._using_tpu}")
        print(f"[ALLOCATE] Input dtype: {self._input_details[0]['dtype']}")
        print(f"[ALLOCATE] Input shape: {self._input_details[0]['shape']}")
        if last_error is not None:
            print(f"[ALLOCATE] TPU init error was: {last_error}")

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

        if self._normalize:
            model_input = padded_image.astype(np.float32) / 255.0
        else:
            model_input = padded_image.astype(self._input_details[0]["dtype"])

        model_input = np.expand_dims(model_input, axis=0)

        self._intepreter.set_tensor(
            self._input_details[0]["index"],
            model_input,
        )
        self._intepreter.invoke()
        raw_out = self._intepreter.get_tensor(self._output_detail[0]["index"])

        if not nms:
            detections = raw_out[0]
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

        out = raw_out[0]

        if out.shape[0] > out.shape[1]:
            predictions = out
        else:
            predictions = out.T

        boxes = predictions[:, :4]
        class_scores = predictions[:, 4:]

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

        x_min = x_center - w / 2
        y_min = y_center - h / 2
        x_max = x_center + w / 2
        y_max = y_center + h / 2

        xyxy_boxes = np.stack([x_min, y_min, x_max, y_max], axis=1)

        indices = cv2.dnn.NMSBoxes(
            xyxy_boxes.tolist(),
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

    def using_tpu(self) -> bool:
        return self._using_tpu