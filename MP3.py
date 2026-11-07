import sys
import os
import traceback
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QMessageBox, QLabel, QProgressBar,
    QComboBox, QFileDialog, QInputDialog, QLineEdit, QGroupBox,
    QStatusBar
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont, QIcon

# --- Пути ---
APP_DIR = Path(__file__).resolve().parent
TOKEN_FILE = APP_DIR / "hf_token.txt"
LOG_FILE = APP_DIR / "transcribe_error.log"
PROMPT_FILE = APP_DIR / "prompt.txt"


# --- Токен Hugging Face (без консоли) ---
def load_hf_token():
    if TOKEN_FILE.exists():
        try:
            token = TOKEN_FILE.read_text(encoding="utf-8").strip()
            if token:
                os.environ["HF_TOKEN"] = token
                return token
        except Exception:
            pass
    return None


def save_hf_token(token: str):
    token = token.strip()
    if token:
        os.environ["HF_TOKEN"] = token
        TOKEN_FILE.write_text(token, encoding="utf-8")
    else:
        os.environ.pop("HF_TOKEN", None)
        if TOKEN_FILE.exists():
            TOKEN_FILE.unlink()


# --- Промпт (подсказка для модели) ---
def load_prompt() -> str:
    if PROMPT_FILE.exists():
        try:
            return PROMPT_FILE.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return ""


def save_prompt(text: str):
    text = text.strip()
    try:
        if text:
            PROMPT_FILE.write_text(text, encoding="utf-8")
        elif PROMPT_FILE.exists():
            PROMPT_FILE.unlink()
    except Exception:
        pass


# --- Импорт тяжёлых библиотек ---
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_OK = True
    FASTER_WHISPER_ERROR = ""
except Exception as e:
    FASTER_WHISPER_OK = False
    FASTER_WHISPER_ERROR = str(e)
    WhisperModel = None


# Глобальные переменные модели
MODEL = None
CURRENT_MODEL_NAME = None


def load_model(model_name: str):
    global MODEL, CURRENT_MODEL_NAME
    if MODEL is None or CURRENT_MODEL_NAME != model_name:
        MODEL = WhisperModel(model_name, device="cpu", compute_type="int8")
        CURRENT_MODEL_NAME = model_name
    return MODEL


class TranscribeWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)  # процент, текущий фрагмент

    def __init__(self, file_path, language, model_name, initial_prompt=""):
        super().__init__()
        self.file_path = file_path
        self.language = language
        self.model_name = model_name
        self.initial_prompt = initial_prompt
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            self.progress.emit(0, "Загрузка модели...")
            load_model(self.model_name)

            self.progress.emit(1, "Анализ аудио...")

            # Промпт передаём только если он не пустой
            kwargs = dict(
                beam_size=5,
                vad_filter=True,
                language=self.language,
                condition_on_previous_text=False,
            )
            if self.initial_prompt.strip():
                kwargs["initial_prompt"] = self.initial_prompt.strip()

            segments, info = MODEL.transcribe(self.file_path, **kwargs)

            total_duration = getattr(info, "duration", 0) or 0
            parts = []
            last_pct = 1

            for segment in segments:
                if self._cancel:
                    self.error.emit("Отменено пользователем.")
                    return

                parts.append(segment.text)

                if total_duration > 0:
                    pct = min(99, int(segment.end / total_duration * 100))
                    if pct > last_pct:
                        last_pct = pct
                        snippet = segment.text.strip()
                        if len(snippet) > 80:
                            snippet = snippet[-80:]
                        self.progress.emit(pct, f"[{segment.end:.1f}s] {snippet}")

            full_text = "".join(parts).strip()

            if not full_text:
                self.error.emit(
                    "Речь не обнаружена. Возможно, файл пуст или содержит только шум."
                )
            else:
                self.progress.emit(100, "Готово")
                self.finished.emit(full_text)

        except Exception as e:
            try:
                LOG_FILE.write_text(traceback.format_exc(), encoding="utf-8")
            except Exception:
                pass
            self.error.emit(
                f"{str(e)}\n\nПодробности сохранены в файле:\n{LOG_FILE}"
            )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Транскрипция аудио (faster-whisper)")
        self.resize(900, 780)

        self.current_file = None
        self.worker = None

        self.setStyleSheet(self._dark_stylesheet())
        self.init_ui()
        self._update_token_status()

    # ---------- UI ----------
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        title = QLabel("Транскрипция аудио и видео")
        f = QFont()
        f.setPointSize(16)
        f.setBold(True)
        title.setFont(f)
        layout.addWidget(title)

        # --- Токен HF ---
        token_box = QGroupBox("Hugging Face токен")
        token_layout = QHBoxLayout(token_box)
        self.token_status = QLabel("")
        token_layout.addWidget(self.token_status, 1)
        btn_token = QPushButton("Настроить токен")
        btn_token.clicked.connect(self.setup_token_dialog)
        token_layout.addWidget(btn_token)
        layout.addWidget(token_box)

        # --- Файл ---
        file_box = QGroupBox("Файл")
        file_layout = QHBoxLayout(file_box)
        self.file_label = QLabel("Файл не выбран")
        self.file_label.setStyleSheet("color: #888;")
        file_layout.addWidget(self.file_label, 1)
        btn_select = QPushButton("Выбрать файл")
        btn_select.clicked.connect(self.select_file)
        file_layout.addWidget(btn_select)
        layout.addWidget(file_box)

        # --- Настройки ---
        settings_box = QGroupBox("Настройки")
        settings_layout = QHBoxLayout(settings_box)

        settings_layout.addWidget(QLabel("Язык:"))
        self.lang_combo = QComboBox()
        self.lang_combo.addItem("Автоопределение", None)
        self.lang_combo.addItem("Русский", "ru")
        self.lang_combo.addItem("Английский", "en")
        self.lang_combo.addItem("Латынь", "la")
        settings_layout.addWidget(self.lang_combo)

        settings_layout.addSpacing(20)

        settings_layout.addWidget(QLabel("Модель:"))
        self.model_combo = QComboBox()
        self.model_combo.addItem("small — быстрая", "small")
        self.model_combo.addItem("medium — точная", "medium")
        self.model_combo.addItem("large-v3 — очень точная (медленно на CPU)", "large-v3")
        settings_layout.addWidget(self.model_combo)

        settings_layout.addStretch()
        layout.addWidget(settings_box)

        # --- Промпт (подсказка для модели) ---
        prompt_box = QGroupBox("Подсказка для модели (initial_prompt) — необязательно")
        prompt_layout = QVBoxLayout(prompt_box)

        hint = QLabel(
            "Слова и фразы через запятую. Помогает распознавать термины, "
            "имена, специфическую лексику. Оставьте пустым, если не уверены — "
            "неверная подсказка ухудшает результат."
        )
        hint.setStyleSheet("color: #999; font-size: 11px;")
        hint.setWordWrap(True)
        prompt_layout.addWidget(hint)

        self.prompt_edit = QTextEdit()
        self.prompt_edit.setFixedHeight(70)
        self.prompt_edit.setPlaceholderText(
            "Пример: Лекция по биологии. Термины: систематика, Геккель, "
            "тайнобрачные, царство растений, эволюция."
        )
        self.prompt_edit.setPlainText(load_prompt())
        self.prompt_edit.setStyleSheet(
            "QTextEdit { font-family: Consolas; font-size: 11px; }"
        )
        prompt_layout.addWidget(self.prompt_edit)

        prompt_btns = QHBoxLayout()
        btn_clear_prompt = QPushButton("Очистить")
        btn_clear_prompt.clicked.connect(self.prompt_edit.clear)
        prompt_btns.addWidget(btn_clear_prompt)
        prompt_btns.addStretch()
        prompt_layout.addLayout(prompt_btns)

        layout.addWidget(prompt_box)

        # --- Кнопки запуска/отмены ---
        actions = QHBoxLayout()
        self.btn_transcribe = QPushButton("Начать транскрипцию")
        self.btn_transcribe.setEnabled(False)
        self.btn_transcribe.clicked.connect(self.start_transcribe)
        actions.addWidget(self.btn_transcribe)

        self.btn_cancel = QPushButton("Отменить")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_transcribe)
        actions.addWidget(self.btn_cancel)

        layout.addLayout(actions)

        # --- Прогресс ---
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        layout.addWidget(self.progress)

        self.status_line = QLabel("Готов к работе")
        self.status_line.setStyleSheet("color: #aaa;")
        self.status_line.setWordWrap(True)
        layout.addWidget(self.status_line)

        # --- Результат ---
        layout.addWidget(QLabel("Результат:"))
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Текст появится здесь...")
        font = QFont("Consolas", 11)
        self.text_edit.setFont(font)
        layout.addWidget(self.text_edit, 1)

        # --- Сохранение ---
        btn_save = QPushButton("Сохранить в TXT")
        btn_save.clicked.connect(self.save_to_txt)
        layout.addWidget(btn_save)

        # --- Статус-бар ---
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Готов к работе")

        # Если faster_whisper не установлен
        if not FASTER_WHISPER_OK:
            self.btn_transcribe.setEnabled(False)
            QMessageBox.critical(
                self, "Ошибка",
                f"Не удалось загрузить faster-whisper:\n{FASTER_WHISPER_ERROR}"
            )

    def _dark_stylesheet(self):
        return """
        QMainWindow, QWidget { background-color: #1e1e1e; color: #ddd; }
        QGroupBox {
            border: 1px solid #333; border-radius: 6px;
            margin-top: 8px; padding-top: 8px;
        }
        QGroupBox::title {
            subcontrol-origin: margin; left: 10px; padding: 0 4px;
            color: #4aa3ff;
        }
        QPushButton {
            background-color: #2d6cdf; color: white; border: none;
            padding: 8px 14px; border-radius: 4px; font-size: 13px;
        }
        QPushButton:hover { background-color: #3d7cef; }
        QPushButton:disabled { background-color: #444; color: #888; }
        QComboBox, QLineEdit {
            background-color: #2a2a2a; color: #eee;
            border: 1px solid #444; border-radius: 4px; padding: 4px 6px;
        }
        QTextEdit {
            background-color: #141414; color: #eee;
            border: 1px solid #333; border-radius: 4px;
        }
        QProgressBar {
            background-color: #222; border: 1px solid #333;
            border-radius: 4px; text-align: center; height: 18px;
        }
        QProgressBar::chunk { background-color: #2d6cdf; border-radius: 3px; }
        QStatusBar { background-color: #181818; color: #999; }
        QLabel { color: #ddd; }
        """

    # ---------- Токен ----------
    def _update_token_status(self):
        if os.environ.get("HF_TOKEN"):
            self.token_status.setText("Токен установлен ✓")
            self.token_status.setStyleSheet("color: #5cb85c;")
        else:
            self.token_status.setText(
                "Токен не задан (скачивание будет анонимным)"
            )
            self.token_status.setStyleSheet("color: #d9a641;")

    def setup_token_dialog(self):
        current = os.environ.get("HF_TOKEN", "")
        info = (
            "Вставьте ваш токен Hugging Face (начинается на hf_...).\n"
            "Если у вас его нет — оставьте поле пустым.\n\n"
            "Токен ускоряет загрузку моделей, но не обязателен.\n"
            "Получить токен: https://huggingface.co/settings/tokens"
        )
        token, ok = QInputDialog.getText(
            self, "Hugging Face токен", info,
            QLineEdit.EchoMode.Normal, current
        )
        if not ok:
            return
        save_hf_token(token)
        self._update_token_status()
        self.statusBar().showMessage("Токен сохранён", 4000)

    # ---------- Файл ----------
    def select_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите медиафайл", "",
            "Медиа файлы (*.mp3 *.mp4 *.wav *.m4a *.avi *.mkv *.ogg *.flac);;"
            "Все файлы (*.*)"
        )
        if path:
            self.current_file = path
            self.file_label.setText(os.path.basename(path))
            self.file_label.setStyleSheet("color: #eee;")
            self.btn_transcribe.setEnabled(True)
            self.statusBar().showMessage(f"Выбран файл: {path}", 5000)

    # ---------- Транскрипция ----------
    def start_transcribe(self):
        if not self.current_file:
            return
        if self.worker is not None and self.worker.isRunning():
            return

        language = self.lang_combo.currentData()
        model_name = self.model_combo.currentData()
        initial_prompt = self.prompt_edit.toPlainText().strip()

        # Сохраняем промпт на будущее
        save_prompt(initial_prompt)

        self.btn_transcribe.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.lang_combo.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.prompt_edit.setEnabled(False)
        self.progress.setValue(0)
        self.text_edit.clear()
        self.status_line.setText("Запуск...")

        self.worker = TranscribeWorker(
            self.current_file, language, model_name, initial_prompt
        )
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.progress.connect(self.on_progress)
        self.worker.start()

    def cancel_transcribe(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.status_line.setText("Отмена...")

    def on_progress(self, pct, msg):
        self.progress.setValue(pct)
        self.status_line.setText(msg)
        self.statusBar().showMessage(msg, 3000)

    def on_finished(self, text):
        self._reset_controls()
        self.progress.setValue(100)
        self.text_edit.setPlainText(text)
        self.status_line.setText("Транскрипция завершена успешно ✓")
        self.statusBar().showMessage("Готово", 4000)

    def on_error(self, error_msg):
        self._reset_controls()
        self.progress.setValue(0)
        self.status_line.setText("Ошибка")
        QMessageBox.critical(
            self, "Ошибка", f"Не удалось обработать файл:\n\n{error_msg}"
        )

    def _reset_controls(self):
        self.btn_transcribe.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.lang_combo.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.prompt_edit.setEnabled(True)
        self.worker = None

    # ---------- Сохранение ----------
    def save_to_txt(self):
        text = self.text_edit.toPlainText()
        if not text.strip():
            QMessageBox.warning(self, "Внимание", "Нет текста для сохранения.")
            return

        default_name = "transcription.txt"
        if self.current_file:
            default_name = Path(self.current_file).stem + ".txt"

        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить как TXT", default_name,
            "Текстовые файлы (*.txt);;Все файлы (*.*)"
        )
        if path:
            try:
                Path(path).write_text(text, encoding="utf-8")
                QMessageBox.information(self, "Успех", f"Сохранено в:\n{path}")
            except Exception as e:
                QMessageBox.critical(
                    self, "Ошибка", f"Не удалось сохранить файл:\n{str(e)}"
                )


# ---------- Точка входа ----------
def main():
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    app = QApplication(sys.argv)
    app.setApplicationName("Whisper Transcribe")

    token = load_hf_token()

    window = MainWindow()
    window.show()

    if not token:
        window.setup_token_dialog()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
