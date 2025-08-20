import sys
from PyQt6.QtWidgets import QApplication
from widgets.main_window import MainWindow
import torch
import traceback

def excepthook(exc_type, exc_value, exc_tb):
        traceback.print_exception(exc_type, exc_value, exc_tb)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sys.exit(1)

if __name__ == "__main__":
    sys.excepthook = excepthook
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    font = app.font()
    font.setPointSize(9)
    app.setFont(font)
    
    try:
        window = MainWindow()
        window.showMaximized()
        sys.exit(app.exec())
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()