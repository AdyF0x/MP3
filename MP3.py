import sys
import os
import traceback
import tkinter as tk
from tkinter import filedialog

# --- Настройка токена Hugging Face через input() ---
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

TOKEN_FILE = "hf_token.txt"
if os.path.exists(TOKEN_FILE):
    # Если файл с токеном уже есть, загружаем его
    with open(TOKEN_FILE, "r", encoding="utf-8") as f:
        os.environ["HF_TOKEN"] = f.read().strip()
    print("--- Токен Hugging Face загружен из файла hf_token.txt ---")
else:
    # Если файла нет, просим ввести токен
    print("\n=== Настройка токена Hugging Face ===")
    print("Если у вас есть токен (начинается на hf_...), вставьте его ниже.")
    print("Если нет, просто нажмите Enter (скачивание будет анонимным и чуть медленнее).\n")
    
    user_token = input("Введите ваш HF_TOKEN: ").strip()
    if user_token:
        os.environ["HF_TOKEN"] = user_token
        # Сохраняем токен в файл для следующих запусков
        with open(TOKEN_FILE, "w", encoding="utf-8") as f:
            f.write(user_token)
        print("--- Токен сохранен в файл hf_token.txt! ---\n")
    else:
        print("--- Токен не введен. Продолжаем без него. ---\n")

# --- Импорт библиотек ---
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QMessageBox, QLabel, QProgressBar, QComboBox
)
from PyQt6.QtCore import QThread, pyqtSignal

from faster_whisper import WhisperModel

# Глобальные переменные
MODEL = None
CURRENT_MODEL_NAME = None

def load_model(model_name):
    """Загружает модель в память, если она еще не загружена или если выбрана другая"""
    global MODEL, CURRENT_MODEL_NAME
    if MODEL is None or CURRENT_MODEL_NAME != model_name:
        print(f"--- Загрузка нейросети ({model_name}) в память... ---")
        MODEL = WhisperModel(model_name, device="cpu", compute_type="int8")
        CURRENT_MODEL_NAME = model_name
        print(f"--- Нейросеть {model_name} успешно загружена ---")


class TranscribeWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, file_path, language, model_name):
        super().__init__()
        self.file_path = file_path
        self.language = language
        self.model_name = model_name

    def run(self):
        try:
            load_model(self.model_name)
            
            print(f"--- Начало транскрипции: {self.file_path} ---")
            print(f"--- Язык: {self.language if self.language else 'Авто'} | Модель: {self.model_name} ---")

            # Подсказка для нейросети
            prompt = "Лекция по биологии и ботанике. Термины: систематика, Геккель, тайнобрачные, царство растений, эволюция, классы, организмы."

            segments, info = MODEL.transcribe(
                self.file_path, 
                beam_size=5, 
                vad_filter=True,
                language=self.language,
                condition_on_previous_text=False,
                initial_prompt=prompt
            )

            full_text = "".join([segment.text for segment in segments]).strip()
            print("--- Транскрипция завершена успешно ---")

            if not full_text:
                self.error.emit("Речь не обнаружена. Возможно, файл пуст или содержит только шум.")
            else:
                self.finished.emit(full_text)

        except Exception as e:
            error_msg = f"Ошибка в потоке: {str(e)}"
            print(f"--- {error_msg} ---")
            
            with open("transcribe_error.log", "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
            
            self.error.emit(f"{str(e)}\n\nПодробности сохранены в файле transcribe_error.log")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Транскрипция (faster-whisper)")
        self.resize(800, 600)

        self.current_file = None
        self.worker = None

        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # --- Выбор файла ---
        file_layout = QHBoxLayout()
        self.file_label = QLabel("Файл не выбран")
        self.file_label.setStyleSheet("color: gray;")
        file_layout.addWidget(self.file_label, 1)

        btn_select = QPushButton("Выбрать файл (MP3/MP4)")
        btn_select.clicked.connect(self.select_file)
        file_layout.addWidget(btn_select)
        layout.addLayout(file_layout)

        # --- Настройки (Язык и Модель) ---
        settings_layout = QHBoxLayout()
        
        settings_layout.addWidget(QLabel("Язык аудио:"))
        self.lang_combo = QComboBox()
        self.lang_combo.addItem("Автоопределение", None)
        self.lang_combo.addItem("Латынь (Latin)", "la")
        self.lang_combo.addItem("Русский (Russian)", "ru")
        self.lang_combo.addItem("Английский (English)", "en")
        settings_layout.addWidget(self.lang_combo)
        
        settings_layout.addSpacing(20)
        
        settings_layout.addWidget(QLabel("Модель:"))
        self.model_combo = QComboBox()
        self.model_combo.addItem("small (Быстрая)", "small")
        self.model_combo.addItem("medium (Точная, медленнее)", "medium")
        settings_layout.addWidget(self.model_combo)
        
        settings_layout.addStretch()
        layout.addLayout(settings_layout)

        # --- Кнопка запуска ---
        self.btn_transcribe = QPushButton("Начать транскрипцию")
        self.btn_transcribe.clicked.connect(self.start_transcribe)
        self.btn_transcribe.setEnabled(False)
        layout.addWidget(self.btn_transcribe)

        # --- Прогресс-бар ---
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        layout.addWidget(self.progress)

        # --- Поле результата ---
        layout.addWidget(QLabel("Результат:"))
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Текст появится здесь...")
        layout.addWidget(self.text_edit, 1)

        # --- Кнопка сохранения ---
        btn_save = QPushButton("Сохранить в TXT")
        btn_save.clicked.connect(self.save_to_txt)
        layout.addWidget(btn_save)

    def select_file(self):
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            path = filedialog.askopenfilename(
                title="Выберите медиафайл",
                filetypes=[("Медиа файлы", "*.mp3 *.mp4 *.wav *.m4a *.avi *.mkv"), ("Все файлы", "*.*")]
            )
            root.destroy()

            if path:
                self.current_file = path
                self.file_label.setText(os.path.basename(path))
                self.file_label.setStyleSheet("color: black;")
                self.btn_transcribe.setEnabled(True)
        except Exception as e:
            print(f"Ошибка при выборе файла: {e}")

    def start_transcribe(self):
        if not self.current_file:
            return

        selected_language = self.lang_combo.currentData()
        selected_model = self.model_combo.currentData()

        self.btn_transcribe.setEnabled(False)
        self.lang_combo.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.progress.show()
        self.text_edit.clear()

        self.worker = TranscribeWorker(self.current_file, selected_language, selected_model)
        self.worker.finished.connect(self.on_transcribe_finished)
        self.worker.error.connect(self.on_transcribe_error)
        self.worker.start()

    def on_transcribe_finished(self, text):
        self.progress.hide()
        self.btn_transcribe.setEnabled(True)
        self.lang_combo.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.text_edit.setPlainText(text)

    def on_transcribe_error(self, error_msg):
        self.progress.hide()
        self.btn_transcribe.setEnabled(True)
        self.lang_combo.setEnabled(True)
        self.model_combo.setEnabled(True)
        QMessageBox.critical(self, "Ошибка", f"Не удалось обработать файл:\n{error_msg}")

    def save_to_txt(self):
        text = self.text_edit.toPlainText()
        if not text.strip():
            QMessageBox.warning(self, "Внимание", "Нет текста для сохранения.")
            return

        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            path = filedialog.asksaveasfilename(
                title="Сохранить как TXT",
                defaultextension=".txt",
                filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")]
            )
            root.destroy()

            if path:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
                QMessageBox.information(self, "Успех", f"Сохранено в:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл:\n{str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    window.raise_()
    window.activateWindow()
    sys.exit(app.exec())
