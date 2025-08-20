from PyQt6.QtWidgets import QLabel, QApplication
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QMouseEvent, QColor, QFont
from PyQt6.QtCore import Qt, QRect, pyqtSignal
import numpy as np
from PIL import Image

class ImageViewer(QLabel):
    """QLabel with rectangle selection functionality"""
    
    selectionMade = pyqtSignal(int, int, int, int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_pixmap = None
        self.original_size = QRect()  
        self.display_rect = QRect()
        self.scale_factor = 1.0
        
        self.selection_rect = QRect()
        self.start_point = None
        self.end_point = None
        self.is_selecting = False
        self.selection_coords = None
        
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: #f5f5f5;
                border: 2px dashed #aaa;
                border-radius: 4px;
            }
        """)
        self.setCursor(Qt.CursorShape.CrossCursor)
    
    def set_image(self, pil_image):
        """Set PIL image and display it"""
        if pil_image is None:
            return
            
        self.original_image = pil_image
        self.original_size = QRect(0, 0, pil_image.width, pil_image.height)
        
        img = pil_image.convert("RGB")
        data = img.tobytes("raw", "RGB")
        q_img = QImage(data, img.size[0], img.size[1], QImage.Format.Format_RGB888)
        self.original_pixmap = QPixmap.fromImage(q_img)
        
        self.selection_coords = None
        self.calculate_display_geometry()
        self.update_display()
    
    def calculate_display_geometry(self):
        """Calculate how the image is displayed within the label"""
        if self.original_pixmap is None:
            return
            
        pixmap_size = self.original_pixmap.size()
        label_size = self.size()
        
        scaled_width = label_size.width()
        scaled_height = int(pixmap_size.height() * (scaled_width / pixmap_size.width()))
        
        if scaled_height > label_size.height():
            scaled_height = label_size.height()
            scaled_width = int(pixmap_size.width() * (scaled_height / pixmap_size.height()))
        
        x = (label_size.width() - scaled_width) // 2
        y = (label_size.height() - scaled_height) // 2
        
        self.display_rect = QRect(x, y, scaled_width, scaled_height)
        
        self.scale_x = scaled_width / pixmap_size.width()
        self.scale_y = scaled_height / pixmap_size.height()
    
    def update_display(self):
        """Update displayed image with current selection"""
        if self.original_pixmap is None:
            return
        
        self.calculate_display_geometry()
        
        scaled_pixmap = self.original_pixmap.scaled(
            self.display_rect.size(),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        final_pixmap = QPixmap(self.size())
        final_pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(final_pixmap)
        
        painter.drawPixmap(self.display_rect, scaled_pixmap)
        
        if not self.selection_rect.isNull() or self.selection_coords:
            self.draw_selection(painter)
        
        painter.end()
        self.setPixmap(final_pixmap)
    
    def draw_selection(self, painter):
        """Draw selection rectangle with style"""
        if self.selection_coords:
            x1, y1, x2, y2 = self.selection_coords
            display_x1 = self.display_rect.x() + int(x1 * self.scale_x)
            display_y1 = self.display_rect.y() + int(y1 * self.scale_y)
            display_x2 = self.display_rect.x() + int(x2 * self.scale_x)
            display_y2 = self.display_rect.y() + int(y2 * self.scale_y)
            
            rect = QRect(
                display_x1, display_y1,
                display_x2 - display_x1, display_y2 - display_y1
            )
        else:
            rect = self.selection_rect
        
        fill_color = QColor(255, 0, 0, 40)
        painter.fillRect(rect, fill_color)
        
        border_pen = QPen(QColor(255, 0, 0), 2, Qt.PenStyle.SolidLine)
        painter.setPen(border_pen)
        painter.drawRect(rect)
        
        handle_size = 6
        for corner in [rect.topLeft(), rect.topRight(), rect.bottomLeft(), rect.bottomRight()]:
            painter.fillRect(
                corner.x() - handle_size//2,
                corner.y() - handle_size//2,
                handle_size, handle_size,
                QColor(255, 0, 0)
            )
        
        if self.selection_coords:
            x1, y1, x2, y2 = self.selection_coords
            text = f"[{x1}, {y1}, {x2}, {y2}]"
            
            font = QFont()
            font.setPointSize(10)
            font.setBold(True)
            painter.setFont(font)
            
            text_rect = painter.fontMetrics().boundingRect(text)
            text_rect.moveTo(rect.x() + 5, rect.y() - text_rect.height() - 5)
            painter.fillRect(text_rect, QColor(0, 0, 0, 180))
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, text)
    
    def mousePressEvent(self, event: QMouseEvent):
        """Start selection on mouse press"""
        if (event.button() == Qt.MouseButton.LeftButton and 
            self.original_pixmap and 
            self.display_rect.contains(event.pos())):
            
            self.is_selecting = True
            self.start_point = event.pos()
            self.selection_rect = QRect()
            self.selection_coords = None
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Update selection during mouse move"""
        if self.is_selecting and self.original_pixmap:
            self.end_point = event.pos()
            self.update_selection_rect()
            self.update_display()
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Finalize selection on mouse release"""
        if event.button() == Qt.MouseButton.LeftButton and self.is_selecting:
            self.is_selecting = False
            
            if not self.selection_rect.isNull():
                img_x1, img_y1, img_x2, img_y2 = self.get_image_coordinates()
                
                img_x1 = max(0, min(img_x1, self.original_size.width() - 1))
                img_y1 = max(0, min(img_y1, self.original_size.height() - 1))
                img_x2 = max(0, min(img_x2, self.original_size.width() - 1))
                img_y2 = max(0, min(img_y2, self.original_size.height() - 1))
                
                img_x1, img_x2 = min(img_x1, img_x2), max(img_x1, img_x2)
                img_y1, img_y2 = min(img_y1, img_y2), max(img_y1, img_y2)
                
                if img_x2 > img_x1 and img_y2 > img_y1:
                    self.selection_coords = (img_x1, img_y1, img_x2, img_y2)
                    self.selectionMade.emit(img_x1, img_y1, img_x2, img_y2)
            
            self.update_display()
    
    def update_selection_rect(self):
        """Update selection rectangle from start and end points"""
        if self.start_point and self.end_point:
            constrained_start = self.constrain_to_display(self.start_point)
            constrained_end = self.constrain_to_display(self.end_point)
            
            self.selection_rect = QRect(constrained_start, constrained_end).normalized()
    
    def constrain_to_display(self, point):
        """Constrain point to image display area"""
        x = max(self.display_rect.left(), min(point.x(), self.display_rect.right()))
        y = max(self.display_rect.top(), min(point.y(), self.display_rect.bottom()))
        return point.__class__(x, y)
    
    def get_image_coordinates(self):
        """Convert display coordinates to original image coordinates"""
        if not self.original_pixmap or self.selection_rect.isNull():
            return 0, 0, 0, 0
        
        x1 = int((self.selection_rect.left() - self.display_rect.x()) / self.scale_x)
        y1 = int((self.selection_rect.top() - self.display_rect.y()) / self.scale_y)
        x2 = int((self.selection_rect.right() - self.display_rect.x()) / self.scale_x)
        y2 = int((self.selection_rect.bottom() - self.display_rect.y()) / self.scale_y)
        
        return x1, y1, x2, y2
    
    def clear_selection(self):
        """Clear current selection"""
        self.selection_rect = QRect()
        self.selection_coords = None
        self.start_point = None
        self.end_point = None
        self.update_display()
    
    def resizeEvent(self, event):
        """Handle resize events to update displayed image"""
        super().resizeEvent(event)
        self.update_display()
    
    def get_selection_coords(self):
        """Get current selection coordinates"""
        return self.selection_coords