from PyQt6.QtCore import QThread, pyqtSignal
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig

class ModelLoaderThread(QThread):
    finished = pyqtSignal(object)  
    error = pyqtSignal(str)       
    progress = pyqtSignal(str)    

    def __init__(self, model_path, attention_type, precision):
        super().__init__()
        self.model_path = model_path
        self.attention_type = attention_type
        self.precision = precision
        self._is_running = True

    def run(self):
        try:
            self.progress.emit("Initializing Florence-2 model...")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if self.precision == "fp16":
                torch_dtype = torch.float16
            elif self.precision == "bf16" and torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32

            self.progress.emit("Loading config...")
            config = AutoConfig.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                attn_implementation=self.attention_type.lower()
            )

            self.progress.emit("Loading model weights...")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                config=config,
                local_files_only=True
            ).to(device)

            self.progress.emit("Loading processor...")
            processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=True
            )

            if self._is_running:
                self.finished.emit({
                    'model': model,
                    'processor': processor,
                    'device': device,
                    'torch_dtype': torch_dtype
                })

        except Exception as e:
            self.error.emit(f"Error loading model: {str(e)}")

    def stop(self):
        self._is_running = False
        self.quit()