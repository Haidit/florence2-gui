from PyQt6.QtWidgets import (
    QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QCheckBox
)
from PyQt6.QtCore import Qt

class GenerationParamsGroup(QGroupBox):
    def __init__(self):
        super().__init__("Generation Parameters")
        layout = QFormLayout()
        layout.setVerticalSpacing(4)
        layout.setHorizontalSpacing(10)
        
        self.max_new_tokens = QSpinBox()
        self.max_new_tokens.setRange(1, 10000)
        self.max_new_tokens.setValue(512)
        layout.addRow("Max tokens:", self.max_new_tokens)

        self.min_new_tokens = QSpinBox()
        self.min_new_tokens.setRange(0, 10000)
        self.min_new_tokens.setValue(0)
        layout.addRow("Min tokens:", self.min_new_tokens)

        self.num_beams = QSpinBox()
        self.num_beams.setRange(1, 10)
        self.num_beams.setValue(1)
        layout.addRow("Num beams:", self.num_beams)

        self.top_k = QSpinBox()
        self.top_k.setRange(1, 1000)
        self.top_k.setValue(50)
        layout.addRow("Top-k:", self.top_k)

        self.no_repeat_ngram_size = QSpinBox()
        self.no_repeat_ngram_size.setRange(0, 10)
        self.no_repeat_ngram_size.setValue(0)
        layout.addRow("No repeat ngram:", self.no_repeat_ngram_size)

        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.1, 5.0)
        self.temperature.setValue(1.0)
        self.temperature.setSingleStep(0.1)
        layout.addRow("Temperature:", self.temperature)

        self.top_p = QDoubleSpinBox()
        self.top_p.setRange(0.0, 1.0)
        self.top_p.setValue(0.9)
        self.top_p.setSingleStep(0.05)
        layout.addRow("Top-p:", self.top_p)

        self.length_penalty = QDoubleSpinBox()
        self.length_penalty.setRange(0.0, 2.0)
        self.length_penalty.setValue(1.0)
        self.length_penalty.setSingleStep(0.1)
        layout.addRow("Length penalty:", self.length_penalty)

        self.repetition_penalty = QDoubleSpinBox()
        self.repetition_penalty.setRange(1.0, 2.0)
        self.repetition_penalty.setValue(1.0)
        self.repetition_penalty.setSingleStep(0.1)
        layout.addRow("Repetition penalty:", self.repetition_penalty)

        self.early_stopping = QCheckBox()
        self.early_stopping.setChecked(False)
        layout.addRow("Early stopping:", self.early_stopping)

        self.do_sample = QCheckBox()
        self.do_sample.setChecked(False)
        layout.addRow("Do sample:", self.do_sample)

        self.setLayout(layout)
        
        self.setStyleSheet("""
            QGroupBox {
                font-size: 11px;
                border: 1px solid #ddd;
                border-radius: 3px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 7px;
                padding: 0 3px;
            }
            QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
                max-height: 22px;
                font-size: 11px;
            }
            QCheckBox {
                spacing: 5px;
                font-size: 11px;
            }
        """)

    def get_params(self):
        return {
            "max_new_tokens": self.max_new_tokens.value(),
            "min_new_tokens": self.min_new_tokens.value(),
            "num_beams": self.num_beams.value(),
            "early_stopping": self.early_stopping.isChecked(),
            "temperature": self.temperature.value(),
            "top_k": self.top_k.value(),
            "top_p": self.top_p.value(),
            "no_repeat_ngram_size": self.no_repeat_ngram_size.value(),
            "length_penalty": self.length_penalty.value(),
            "repetition_penalty": self.repetition_penalty.value(),
            "do_sample": self.do_sample.isChecked(),
        }