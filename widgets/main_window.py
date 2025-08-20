from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QLineEdit, QGroupBox, QScrollArea, QFileDialog,
    QTextEdit, QFormLayout, QProgressBar, QApplication, QSpinBox
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, pyqtSignal
import cv2
import torch
from PIL import Image, ImageDraw
import numpy as np
import os

from config import LOCAL_MODELS, TASK_TAGS
from core.model_wrapper import Florence2ModelWrapper
from core.video_processor import BatchVideoProcessor
from threads.model_loader import ModelLoaderThread
from threads.processing import ProcessingThread
from threads.video_processing import VideoProcessingThread
from widgets.params_group import GenerationParamsGroup

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model_loader = None
        self.processing_thread = None
        self.florence_model = None
        self.image_pil = None
        self.full_result_image = None
        self.video_processor = None
        self.video_thread = None
        self.video_input_path = ""
        self.video_output_path = ""

        self.setWindowTitle("Florence-2 GUI")
        self.setMinimumSize(1400, 900)
        
        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(10)
        
        # Left Panel (Controls) - 30% width
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)
        left_panel.setContentsMargins(0, 0, 0, 0)
        main_layout.addLayout(left_panel, stretch=3)

        # Right Panel (Images) - 70% width
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)
        main_layout.addLayout(right_panel, stretch=7)

        # 1. Model Settings Group
        model_group = QGroupBox("Model Settings")
        model_layout = QFormLayout()
        model_layout.setVerticalSpacing(5)
        
        self.model_combo = QComboBox()
        self.model_combo.clear()
        for model_name in LOCAL_MODELS:
            self.model_combo.addItem(model_name)       
        
        self.attention_combo = QComboBox()
        self.attention_combo.addItems(["sdpa", "eager"]) #TODO: install flash_attn
        
        self.precision_combo = QComboBox()
        self.precision_combo.addItems(["fp32", "fp16", "bf16"])
        
        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.setMaximumHeight(28)
        
        model_layout.addRow("Model:", self.model_combo)
        model_layout.addRow("Attention:", self.attention_combo)
        model_layout.addRow("Precision:", self.precision_combo)
        model_layout.addRow(self.load_model_btn)
        model_group.setLayout(model_layout)
        left_panel.addWidget(model_group)

        # 2. Request Settings Group
        request_group = QGroupBox("Request Settings")
        request_layout = QFormLayout()
        request_layout.setVerticalSpacing(5)
        
        self.category_combo = QComboBox()
        self.category_combo.addItems(list(TASK_TAGS.keys()))
        
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Enter prompt...")
        
        request_layout.addRow("Category:", self.category_combo)
        request_layout.addRow("Prompt:", self.prompt_input)
        request_group.setLayout(request_layout)
        left_panel.addWidget(request_group)

        # 3. Generation Parameters (with scroll)
        params_scroll = QScrollArea()
        params_scroll.setWidgetResizable(True)
        params_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.generation_params = GenerationParamsGroup()
        params_scroll.setWidget(self.generation_params)
        left_panel.addWidget(params_scroll)

        # 4. Image Upload
        upload_group = QGroupBox("Image Input")
        upload_layout = QVBoxLayout()
        upload_layout.setSpacing(8)
        
        self.upload_btn = QPushButton("📁 Upload Image")
        self.upload_btn.setMaximumHeight(28)
        upload_layout.addWidget(self.upload_btn)
        
        self.upload_info = QLabel("No image loaded")
        self.upload_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.upload_info.setStyleSheet("font-size: 11px; color: #666;")
        upload_layout.addWidget(self.upload_info)
        
        upload_group.setLayout(upload_layout)
        left_panel.addWidget(upload_group)

        # 5. Video Processing Settings Group
        video_group = QGroupBox("Video Processing Settings")
        video_layout = QFormLayout()
        video_layout.setVerticalSpacing(5)

        # Batch size control
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 16) 
        self.batch_size_spin.setValue(4)
        video_layout.addRow("Batch size:", self.batch_size_spin)

        # Frame skip control
        self.frame_skip_spin = QSpinBox()
        self.frame_skip_spin.setRange(1, 10)
        self.frame_skip_spin.setValue(2) 
        video_layout.addRow("Frame skip:", self.frame_skip_spin)

        # Video file selection buttons
        video_btn_layout = QHBoxLayout()
        self.video_input_btn = QPushButton("📹 Select Video")
        self.video_input_btn.setMaximumHeight(28)
        self.process_video_btn = QPushButton("🎬 Process Video")
        self.process_video_btn.setMaximumHeight(28)
        self.process_video_btn.setEnabled(False)
        video_btn_layout.addWidget(self.video_input_btn)
        video_btn_layout.addWidget(self.process_video_btn)
        video_layout.addRow(video_btn_layout)

        # Progress bar
        self.video_progress = QProgressBar()
        self.video_progress.setTextVisible(False)
        video_layout.addRow(self.video_progress)

        # Video info label
        self.video_info = QLabel("No video selected")
        self.video_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_info.setStyleSheet("font-size: 11px; color: #666;")
        video_layout.addRow(self.video_info)

        video_group.setLayout(video_layout)
        left_panel.addWidget(video_group)

        # Image Display Area
        image_display_layout = QHBoxLayout()
        image_display_layout.setSpacing(15)
        
        # Input Image
        self.input_image = QLabel()
        self.input_image.setMinimumSize(600, 500)
        self.input_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_image.setStyleSheet("""
            QLabel {
                background-color: #f5f5f5;
                border: 2px dashed #aaa;
                border-radius: 4px;
            }
        """)
        self.input_image.setCursor(Qt.CursorShape.PointingHandCursor)
        self.input_image.mousePressEvent = self.show_original_input_image
        image_display_layout.addWidget(self.input_image)

        # Output Image
        output_images_layout = QVBoxLayout()
        output_images_layout.setSpacing(15)
        
        self.result_image = QLabel()
        self.result_image.setMinimumSize(600, 500)
        self.result_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_image.setStyleSheet("""
            QLabel {
                background-color: #f5f5f5;
                border: 2px dashed #aaa;
                border-radius: 4px;
            }
        """)
        self.result_image.setCursor(Qt.CursorShape.PointingHandCursor)
        self.result_image.mousePressEvent = self.show_original_result_image
        output_images_layout.addWidget(self.result_image)
        image_display_layout.addLayout(output_images_layout)
        right_panel.addLayout(image_display_layout)

        # Results Log
        self.result_text = QTextEdit()
        self.result_text.setMaximumHeight(120)
        self.result_text.setStyleSheet("""
            QTextEdit {
                font-family: Consolas, monospace;
                font-size: 11px;
                border: 1px solid #ddd;
                border-radius: 3px;
            }
        """)
        right_panel.addWidget(self.result_text)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        self.run_btn = QPushButton("🚀 Run Processing")
        self.run_btn.setStyleSheet("""
        QPushButton {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            padding: 8px;
            border-radius: 4px;
            font-size: 12px;
        }
        QPushButton:disabled {
            background-color: #cccccc;
        }
        QPushButton:hover:!disabled {
            background-color: #45a049;
        }
        """)
        self.run_btn.setMaximumHeight(40)

        self.save_btn = QPushButton("💾 Save Result")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        self.save_btn.setMaximumHeight(40)

        self.cancel_btn = QPushButton("✖ Cancel")
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            QPushButton:hover:!disabled {
                background-color: #d32f2f;
            }
        """)
        self.cancel_btn.setMaximumHeight(40)
        self.cancel_btn.setEnabled(False)

        button_layout.addWidget(self.run_btn)
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.save_btn)
        
        right_panel.addLayout(button_layout)

        # Connect signals
        self.load_model_btn.clicked.connect(self.start_model_loading)
        self.upload_btn.clicked.connect(self.upload_image)
        self.run_btn.clicked.connect(self.run_processing)
        self.save_btn.clicked.connect(self.save_output_image)
        self.cancel_btn.clicked.connect(self.cancel_processing)
        self.video_input_btn.clicked.connect(self.select_video_file)
        self.process_video_btn.clicked.connect(self.start_video_processing)

    def start_model_loading(self):
        if self.model_loader and self.model_loader.isRunning():
            self.log_message("\nStopping previous model loading...")
            try:
                self.model_loader.finished.disconnect()
                self.model_loader.error.disconnect()
                self.model_loader.progress.disconnect()
            except:
                pass
            
            self.model_loader.requestInterruption()
            self.model_loader.quit() 
            self.model_loader.wait(2000)
            
            if self.model_loader.isRunning():
                self.log_message("Warning: Thread did not stop gracefully!")
                self.model_loader.terminate()
                self.model_loader.wait()
            
            self.model_loader = None

            self.model_loader.stop()
            
        model_name = self.model_combo.currentText()
        if model_name not in LOCAL_MODELS:
            self.log_message(f"\nError: Model {model_name} not found locally!")
            return
        
        model_path = LOCAL_MODELS[model_name]["path"]
        attention_type = self.attention_combo.currentText()
        precision = self.precision_combo.currentText()
        
        self.log_message("\nStarting model loading from local storage...")
        self.load_model_btn.setEnabled(False)
        self.load_model_btn.setText("Loading...")
        
        self.model_loader = ModelLoaderThread(model_path, attention_type, precision)
        self.model_loader.finished.connect(self.on_model_loaded)
        self.model_loader.error.connect(self.on_model_error)
        self.model_loader.progress.connect(self.log_message)
        self.model_loader.start()
 
    def on_model_loaded(self, model_data):
        self.florence_model = Florence2ModelWrapper(model_data)
        self.log_message("Model successfully loaded!")
        self.load_model_btn.setEnabled(True)
        self.load_model_btn.setText("Load Model")
        # self.model_loader = None

    def on_model_error(self, error_msg):
        self.log_message(error_msg)
        self.load_model_btn.setEnabled(True)
        self.load_model_btn.setText("Load Model")
        # self.model_loader = None
           
    def closeEvent(self, event):
        if self.model_loader and self.model_loader.isRunning():
            self.model_loader.stop()
            self.model_loader.wait()

        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.processing_thread.wait()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if hasattr(self, 'florence_model') and self.florence_model:
            del self.florence_model.model
            del self.florence_model.processor
            self.florence_model = None

        for attr in ['image_pil', 'full_result_image']:
            if hasattr(self, attr):
                delattr(self, attr)

        import gc
        gc.collect()

        super().closeEvent(event)

    def upload_image(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Image", 
            "", 
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if fname:
            img_cv2 = cv2.imread(fname)
            if img_cv2 is not None:
                img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
                self.image_pil = Image.fromarray(img_cv2)
                h, w, _ = img_cv2.shape
                
                q_img = QImage(
                    img_cv2.data, 
                    w, 
                    h, 
                    img_cv2.strides[0], 
                    QImage.Format.Format_RGB888
                )
                
                pixmap = QPixmap.fromImage(q_img)
                scaled_pixmap = pixmap.scaled(
                    self.input_image.size(), 
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                
                self.input_image.setPixmap(scaled_pixmap)
                self.upload_info.setText(f"Loaded: {w}x{h} px")
                self.log_message(f"Image loaded: {fname} ({w}x{h} px)")
            else:
                self.log_message("Error: Failed to load image")

    def run_processing(self):
        if not self.florence_model:
            self.log_message("\nError: Model not loaded!")
            return
            
        if not self.image_pil:
            self.log_message("\nError: No image loaded!")
            return
        
        if self.processing_thread and self.processing_thread.isRunning():
            self.log_message("\nError: Processing already in progress!")
            return
        
        params = self.generation_params.get_params()
        prompt = self.prompt_input.text()
        selected_category = self.category_combo.currentText()
        task_tag = TASK_TAGS.get(selected_category, "<CAPTION>")
        
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Processing...")
        self.save_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        
        self.log_message("\n=== Starting Processing ===")
        self.log_message(f"Task: {selected_category}")
        self.log_message(f"Prompt: {prompt}")
        
        self.processing_thread = ProcessingThread(
            model_wrapper=self.florence_model,
            image_pil=self.image_pil,
            task_tag=task_tag,
            prompt=prompt,
            params=params
        )
        
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.error.connect(self.on_processing_error)
        self.processing_thread.status.connect(self.log_message)
        
        self.processing_thread.start()
    
    def on_processing_finished(self, result):
        self.display_results(result)
        self.run_btn.setEnabled(True)
        self.run_btn.setText("🚀 Run Processing")
        self.save_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.processing_thread = None

    def on_processing_error(self, error_msg):
        self.log_message(f"\n{error_msg}")
        self.run_btn.setEnabled(True)
        self.run_btn.setText("🚀 Run Processing")
        self.save_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        self.processing_thread = None

    def cancel_processing(self):
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.log_message("\nProcessing canceled by user")
            self.run_btn.setEnabled(True)
            self.run_btn.setText("🚀 Run Processing")
            self.save_btn.setEnabled(False)
            self.cancel_btn.setEnabled(False)

    def display_results(self, result):
        self.log_message(f"Result: {str(result)}")
        self.log_message("\n=== Processing Results ===")
        self.log_message(f"Task type: {result['task_tag']}")
        self.log_message(f"Generated text:\n{result['text']}")
        
        self.result_image.clear()
        self.full_result_image = None

        if result['is_visual_task'] and 'processed_results' in result:
            annotated_img = self._visualize_results(
                self.image_pil, 
                result['processed_results'], 
                result['task_tag']
            )
            
            self.full_result_image = annotated_img.copy()
            self._display_pil_image(annotated_img, self.result_image)
        else:
            self.result_image.setText(self._format_text_for_display(result['text']))
            self.result_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
    
    def _visualize_results(self, image_pil, processed_results, task_tag=None):
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        if not processed_results or not next(iter(processed_results.values())):
            return image_pil
        
        result_data = next(iter(processed_results.values()))
        palette = [
            (0, 255, 0), (0, 0, 255), (255, 0, 0),
            (0, 255, 255), (255, 0, 255), (255, 255, 0)
        ]
        
        if 'quad_boxes' in result_data and 'labels' in result_data:
            for i, (quad_box, label) in enumerate(zip(
                result_data['quad_boxes'],
                result_data['labels']
            )):
                color = palette[i % len(palette)]
                
                pts = np.array([
                    [quad_box[0], quad_box[1]],
                    [quad_box[2], quad_box[3]],
                    [quad_box[4], quad_box[5]],
                    [quad_box[6], quad_box[7]]
                ], dtype=np.int32)
                
                cv2.polylines(image_cv, [pts], isClosed=True, color=color, thickness=2)
                
                if label:
                    text_x = int(quad_box[0])
                    text_y = int(quad_box[1]) - 10
                    if text_y < 0:
                        text_y = int(quad_box[7]) + 20
                    
                    cv2.putText(image_cv, label, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if 'bboxes' in result_data:
            labels = result_data.get('bboxes_labels', result_data.get('labels', [''] * len(result_data['bboxes'])))
            
            for i, (bbox, label) in enumerate(zip(result_data['bboxes'], labels)):
                x1, y1, x2, y2 = map(int, bbox)
                color = palette[i % len(palette)]
                cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image_cv, label, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if 'polygons' in result_data:
            for i, polygon in enumerate(result_data['polygons']):
                color = palette[i % len(palette)]
                pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
                cv2.polylines(image_cv, [pts], isClosed=True, color=color, thickness=2)
                
                overlay = image_cv.copy()
                cv2.fillPoly(overlay, [pts], color)
                alpha = 0.3
                cv2.addWeighted(overlay, alpha, image_cv, 1 - alpha, 0, image_cv)
        
        return Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    
    def _display_pil_image(self, pil_img, qlabel):
        img = pil_img.convert("RGB")
        data = img.tobytes("raw", "RGB")
        q_img = QImage(data, img.size[0], img.size[1], QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        scaled_pixmap = pixmap.scaled(
            qlabel.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        qlabel.setPixmap(scaled_pixmap)

    def _format_text_for_display(self, text, max_line_length=80):
        cleaned_text = text.replace('</s>', '').replace('<s>', '').strip()
        words = cleaned_text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line) + len(word) + 1 > max_line_length:
                lines.append(current_line)
                current_line = word
            else:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
                    
        if current_line:
            lines.append(current_line)
            
        return "\n".join(lines)

    def save_output_image(self):
        if self.full_result_image is None:
            self.log_message("Error: No image to save!")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить изображение",
            "",
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;BMP Image (*.bmp)"
        )
        
        if file_path:
            try:
                if file_path.lower().endswith(('.jpg', '.jpeg')):
                    format = 'JPEG'
                    quality = 95
                    self.full_result_image.save(file_path, format, quality=quality)
                elif file_path.lower().endswith('.bmp'):
                    format = 'BMP'
                    self.full_result_image.save(file_path, format)
                else:
                    format = 'PNG'
                    self.full_result_image.save(file_path, format)
                
                self.log_message(f"\nImage saved: {file_path}")
            except Exception as e:
                self.log_message(f"\nSaving image error: {str(e)}")

    def log_message(self, message):
        self.result_text.append(message)
        self.result_text.verticalScrollBar().setValue(
            self.result_text.verticalScrollBar().maximum()
        )

    def show_original_input_image(self, _):
        if self.image_pil:
            self._show_original_image(self.image_pil, "Original Input Image")

    def show_original_result_image(self, _):
        if self.full_result_image:
            self._show_original_image(self.full_result_image, "Original Result Image")

    def _show_original_image(self, pil_image, title):
        image_window = QMainWindow(self)
        image_window.setWindowTitle(title)
        
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        if pil_image.mode == 'RGBA':
            q_img = QImage(
                pil_image.tobytes('raw', 'RGBA'), 
                pil_image.width, 
                pil_image.height, 
                QImage.Format.Format_RGBA8888
            )
        else:
            q_img = QImage(
                pil_image.tobytes('raw', 'RGB'), 
                pil_image.width, 
                pil_image.height, 
                pil_image.width * 3, 
                QImage.Format.Format_RGB888
            )
        
        pixmap = QPixmap.fromImage(q_img)
        
        image_label.setPixmap(pixmap)
        
        scroll_area = QScrollArea()
        scroll_area.setWidget(image_label)
        scroll_area.setWidgetResizable(True)
        
        image_window.setCentralWidget(scroll_area)
        
        screen_size = QApplication.primaryScreen().availableSize()
        image_window.resize(
            int(screen_size.width() * 0.8), 
            int(screen_size.height() * 0.8)
        )
        
        image_window.show()

    def select_video_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Выберите видео файл", 
            "", 
            "Видео файлы (*.mp4 *.mov *.avi *.mkv)"
        )
        
        if file_path:
            self.video_input_path = file_path
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps
                self.video_info.setText(
                    f"{os.path.basename(file_path)}\n"
                    f"{frame_count} кадров, {fps:.1f} FPS, {duration:.1f} сек"
                )
                self.process_video_btn.setEnabled(True)
                cap.release()
            else:
                self.video_info.setText("Ошибка чтения видео")
                self.process_video_btn.setEnabled(False)

    def start_video_processing(self):
        if not self.florence_model:
            self.log_message("\nError: Model not loaded!")
            return
        
        if not self.video_input_path:
            return
            
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить обработанное видео",
            "",
            "MP4 files (*.mp4)"
        )
        
        if not output_path:
            return
            
        batch_size = self.batch_size_spin.value()
        frame_skip = self.frame_skip_spin.value()
        generation_params = self.generation_params.get_params()
        task_tag = TASK_TAGS.get(self.category_combo.currentText(), "<CAPTION>")
        prompt = self.prompt_input.text()
        
        self.video_processor = BatchVideoProcessor(self.florence_model)
        self.video_processor.update_parameters(batch_size, frame_skip)
        
        self.log_message("\n=== Starting Video Processing ===")
        self.log_message(f"Input: {self.video_input_path}")
        self.log_message(f"Output: {output_path}")
        self.log_message(f"Batch size: {batch_size}, Frame skip: {frame_skip}")
        self.log_message(f"Task: {task_tag}, Prompt: {prompt}")
        
        self.video_thread = VideoProcessingThread(
            self.video_processor,
            self.video_input_path,
            output_path,
            task_tag,
            prompt,
            generation_params
        )
        
        self.video_thread.progress_updated.connect(self.update_video_progress)
        self.video_thread.finished_signal.connect(lambda success, msg, time: self.video_processing_finished(success, msg, time))
        self.video_thread.start()
        
        self.process_video_btn.setEnabled(False)
        self.video_input_btn.setEnabled(False)

    def update_video_progress(self, value):
        self.video_progress.setValue(value)

    def video_processing_finished(self, success, message, processing_time):
        self.process_video_btn.setEnabled(True)
        self.video_input_btn.setEnabled(True)

        minutes, seconds = divmod(processing_time, 60)
        time_str = f"{minutes} minutes {int(seconds)} seconds"

        if success:
            self.log_message(f"\nVideo processing completed successfully in {time_str}!")
            self.video_progress.setValue(100)
        else:
            self.log_message(f"\nVideo processing failed: {message}")
            self.video_progress.setValue(0)
