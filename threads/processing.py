from PyQt6.QtCore import QThread, pyqtSignal

class ProcessingThread(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    status = pyqtSignal(str) 

    def __init__(self, model_wrapper, image_pil, task_tag, prompt, params):
        super().__init__()
        self.model_wrapper = model_wrapper
        self.image_pil = image_pil
        self.task_tag = task_tag
        self.prompt = prompt
        self.params = params
        self._is_running = True

    def run(self):
        try:
            self.status.emit("Starting image processing...")
            
            result = self.model_wrapper.process_image(
                image_pil=self.image_pil,
                task_tag=self.task_tag,
                prompt=self.prompt,
                generation_params=self.params
            )
            
            if self._is_running:
                self.finished.emit(result)
                self.status.emit("Processing completed!")
                
        except Exception as e:
            self.error.emit(f"Processing error: {str(e)}")

    def stop(self):
        self._is_running = False
        self.quit()