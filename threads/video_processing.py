from PyQt6.QtCore import QThread, pyqtSignal

class VideoProcessingThread(QThread):
    progress_updated = pyqtSignal(int)
    finished_signal = pyqtSignal(bool, str, float)
    
    def __init__(self, processor, input_path, output_path, task_tag, prompt, generation_params):
        super().__init__()
        self.processor = processor
        self.input_path = input_path
        self.output_path = output_path
        self.task_tag = task_tag
        self.prompt = prompt
        self.generation_params = generation_params
        
    def run(self):
        try:
            success, processing_time = self.processor.process_video(
                self.input_path,
                self.output_path,
                self.task_tag,
                self.prompt,
                self.generation_params,
                self.progress_updated.emit
            )
            self.finished_signal.emit(success, self.output_path, processing_time)
        except Exception as e:
            print(f"Ошибка обработки видео: {str(e)}")
            self.finished_signal.emit(False, str(e), 0.0)
