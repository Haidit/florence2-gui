import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
from collections import deque

BATCH_SIZE = 4
FRAME_SKIP = 20
MODEL_NAME = "./models/florence-2-base"
PROMPT = "<OD>"

class BatchVideoProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if "cuda" in self.device else torch.float32

        print("Инициализация модели...")
        self.config = AutoConfig.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            attn_implementation="eager"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            config=self.config,
            local_files_only=True
        ).to(self.device).eval()
        
        self.processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            local_files_only=True
        )
        
        self.frame_buffer = deque(maxlen=BATCH_SIZE)
        self.original_frames = []

    def process_batch(self):
        if not self.frame_buffer:
            return []

        inputs = self.processor(
            text=[PROMPT] * len(self.frame_buffer),
            images=list(self.frame_buffer),
            return_tensors="pt",
            padding=True
        ).to(self.device, self.torch_dtype)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                num_beams=1,
                do_sample=False,
                use_cache=False,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )

        results = []
        for i in range(len(self.frame_buffer)):
            generated_text = self.processor.batch_decode([outputs[i]], skip_special_tokens=False)[0]
            result = self.processor.post_process_generation(
                generated_text,
                task=PROMPT.split('>')[0] + '>',
                image_size=self.frame_buffer[i].size
            )
            results.append(result.get(PROMPT.split('>')[0] + '>', {}))
        
        return results
    
    def _process_remaining(self, out):
        results = self.process_batch()
        self._write_results(out, results)

    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Не удалось открыть видео: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps/FRAME_SKIP, (width, height))

        print(f"Обработка видео (батч={BATCH_SIZE}, пропуск={FRAME_SKIP} кадров)...")
        
        pbar = tqdm(total=int(total_frames/FRAME_SKIP))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                if self.frame_buffer:
                    results = self.process_batch()
                    self._write_results(out, results)
                    pbar.update(len(results))
                break

            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % FRAME_SKIP != 0:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            
            self.frame_buffer.append(img_pil)
            self.original_frames.append(frame.copy())

            if len(self.frame_buffer) == BATCH_SIZE:
                results = self.process_batch()
                self._write_results(out, results)
                pbar.update(len(results))
                self.frame_buffer.clear()
                self.original_frames.clear()

        cap.release()
        out.release()
        pbar.close()
        print(f"Обработка завершена. Результат сохранен в {output_path}")

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


if __name__ == "__main__":
    processor = BatchVideoProcessor()
    processor.process_video("././videos/IMG_5192.MOV", "output1.mp4")