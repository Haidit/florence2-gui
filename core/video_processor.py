import cv2
from collections import deque
import time
from PIL import Image

class BatchVideoProcessor:
    def __init__(self, model_wrapper):
        self.model_wrapper = model_wrapper
        self.frame_buffer = deque()
        self.original_frames = []
        self.batch_size = 4 
        self.frame_skip = 2 

    def update_parameters(self, batch_size, frame_skip):
        self.batch_size = batch_size
        self.frame_skip = frame_skip
        self.frame_buffer = deque(maxlen=batch_size)

    def process_batch(self, task_tag, prompt, generation_params):
        if not self.frame_buffer:
            return []

        try:
            results = []
            for img_pil in self.frame_buffer:
                result = self.model_wrapper.process_image(
                    image_pil=img_pil,
                    task_tag=task_tag,
                    prompt=prompt,
                    generation_params=generation_params
                )
                
                if result['success'] and result['is_visual_task']:
                    results.append(result['processed_results'].get(task_tag, {}))
                else:
                    results.append({})
            
            return results
        except Exception as e:
            print(f"Ошибка обработки батча: {str(e)}")
            return []

    def process_video(self, input_path, output_path, task_tag, prompt, generation_params, progress_callback=None):
        start_time = time.time()
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Не удалось открыть видео: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps/max(1, self.frame_skip), (width, height))

        processed_frames = 0
        frame_count = 0
        self.frame_buffer.clear()
        self.original_frames.clear()

        while True:
            ret, frame = cap.read()
            if not ret:
                if self.frame_buffer:
                    results = self.process_batch(task_tag, prompt, generation_params)
                    self._write_results(out, results)
                    processed_frames += len(results)
                    if progress_callback:
                        progress = int((processed_frames / (total_frames/max(1, self.frame_skip)) * 100))
                        progress_callback(progress)
                break

            frame_count += 1
            
            if self.frame_skip > 1 and frame_count % self.frame_skip != 0:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            
            self.frame_buffer.append(img_pil)
            self.original_frames.append(frame.copy())

            if len(self.frame_buffer) >= self.batch_size:
                results = self.process_batch(task_tag, prompt, generation_params)
                self._write_results(out, results)
                processed_frames += len(results)
                if progress_callback:
                    progress = int((processed_frames / (total_frames/max(1, self.frame_skip)) * 100))
                    progress_callback(progress)
                self.frame_buffer.clear()
                self.original_frames.clear()

        cap.release()
        out.release()

        processing_time = time.time() - start_time
        return True, processing_time

    def _write_results(self, out, results):
        for i, result in enumerate(results):
            if i >= len(self.original_frames):
                continue
                
            frame = self.original_frames[i]
            
            if result and "bboxes" in result:
                for j, bbox in enumerate(result["bboxes"]):
                    x1, y1, x2, y2 = map(int, bbox)
                    label = result.get("labels", [""])[j] if "labels" in result else ""
                    
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            out.write(frame)
