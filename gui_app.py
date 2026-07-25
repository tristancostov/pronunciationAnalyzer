import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import json
import threading
import numpy as np
import wave
from datetime import datetime

try:
    import pronunciationAnalyzer as pa
except ImportError:
    messagebox.showerror("Ошибка",
        "Не найден файл pronunciationAnalyzer.py! Убедитесь, что он в той же папке.")

# matplotlib для графиков
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt

# Настройка кириллического шрифта
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']


class PronunciationGUI:
    TRAINING_PHRASES = (
        "Сегодня хорошая погода.",
        "Русская речь требует точного ударения.",
        "Я внимательно слушаю и повторяю фразу.",
        "Организация проводит интересное исследование.",
    )

    def __init__(self, root):
        self.root = root
        self.root.title("Анализатор произношения — система оценки русской речи")
        self.root.geometry("1400x850")

        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Treeview.Heading", font=("Arial", 10, "bold"))
        style.configure("Treeview", font=("Arial", 10), rowheight=26)

        self.audio_path = None
        self.analysis_data = None
        self.full_signal = None
        self.sr = 16000
        self.models = None       # (voskModel, KaldiRecognizer, morphAnalyzer, accentizer)
        self.analysis_audio_path = None
        self.pending_reference = ""
        self.pending_asr = "whisper"
        self.record_stream = None
        self.record_chunks = []
        self.record_sample_rate = 0

        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Загружаем модели в фоне при старте
        self.lbl_status.config(text="Загрузка моделей (VOSK, ruaccent)…", fg="orange")
        threading.Thread(target=self._preload_models, daemon=True).start()

    # ═══════════════════════════════════════════════════════════
    #  ПОСТРОЕНИЕ ИНТЕРФЕЙСА
    # ═══════════════════════════════════════════════════════════
    def create_widgets(self):
        # --- Верхняя панель ---
        top_frame = tk.Frame(self.root, pady=8, padx=12)
        top_frame.pack(fill=tk.X)

        self.btn_load = tk.Button(top_frame, text="📁 Выбрать аудио",
                                  font=("Arial", 11), command=self.load_audio)
        self.btn_load.pack(side=tk.LEFT, padx=4)

        self.btn_record = tk.Button(top_frame, text="🎙 Записать",
                                    font=("Arial", 11),
                                    command=self.toggle_recording)
        self.btn_record.pack(side=tk.LEFT, padx=4)

        self.lbl_file = tk.Label(top_frame, text="Файл не выбран",
                                 font=("Arial", 11), fg="gray")
        self.lbl_file.pack(side=tk.LEFT, padx=12)

        self.btn_analyze = tk.Button(top_frame, text="▶ Анализировать",
                                     font=("Arial", 11, "bold"),
                                     bg="#4CAF50", fg="white",
                                     command=self.start_analysis,
                                     state=tk.DISABLED)
        self.btn_analyze.pack(side=tk.RIGHT, padx=4)

        # --- Режим работы ---
        mode_frame = tk.Frame(self.root, pady=3, padx=12)
        mode_frame.pack(fill=tk.X)
        tk.Label(mode_frame, text="Режим:",
                 font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value="free")
        tk.Radiobutton(mode_frame, text="Свободная речь (без текста)",
                       variable=self.mode_var, value="free",
                       command=self._on_mode_changed).pack(side=tk.LEFT, padx=8)
        tk.Radiobutton(mode_frame, text="Тренировка по фразе приложения",
                       variable=self.mode_var, value="guided",
                       command=self._on_mode_changed).pack(side=tk.LEFT, padx=8)
        tk.Label(mode_frame, text="Распознавание:",
                 font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(24, 4))
        self.asr_var = tk.StringVar(value="Whisper — высокая точность")
        self.cmb_asr = ttk.Combobox(
            mode_frame, textvariable=self.asr_var, state="readonly", width=24,
            values=("Whisper — высокая точность", "VOSK — быстрее"))
        self.cmb_asr.pack(side=tk.LEFT)
        self.cmb_asr.current(0)

        # --- Фраза приложения (только для направленной тренировки) ---
        text_frame = tk.Frame(self.root, pady=3, padx=12)
        text_frame.pack(fill=tk.X)

        tk.Label(text_frame, text="Фраза для тренировки:",
                font=("Arial", 10, "bold")).pack(anchor=tk.W)

        self.cmb_phrase = ttk.Combobox(text_frame,
                                       values=self.TRAINING_PHRASES,
                                       state=tk.DISABLED,
                                       font=("Arial", 10))
        self.cmb_phrase.pack(fill=tk.X, pady=(3, 0))
        self.cmb_phrase.bind("<<ComboboxSelected>>", self._select_phrase)

        self.txt_reference = tk.Text(text_frame, height=3,
                                     font=("Arial", 11), wrap=tk.WORD)
        self.txt_reference.pack(fill=tk.X, pady=3)
        self.txt_reference.config(state=tk.DISABLED, background="#eeeeee")

        # --- Основная область: таблица слева + графики справа ---
        main_pw = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashwidth=6)
        main_pw.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # == Левая панель: таблица слов ==
        left_frame = tk.Frame(main_pw, width=580)
        main_pw.add(left_frame, stretch="always")

        columns = ("word", "syllables", "stress", "reduction", "issue")
        self.tree = ttk.Treeview(left_frame, columns=columns,
                                 show="headings", height=22)
        self.tree.heading("word", text="Слово")
        self.tree.heading("syllables", text="Ядра V2 / V1 / текст")
        self.tree.heading("stress", text="Ударение (акуст. vs словарь)")
        self.tree.heading("reduction", text="Редукция")
        self.tree.heading("issue", text="Сигнал")

        self.tree.column("word", width=100, anchor=tk.W)
        self.tree.column("syllables", width=145, anchor=tk.CENTER)
        self.tree.column("stress", width=190, anchor=tk.CENTER)
        self.tree.column("reduction", width=80, anchor=tk.CENTER)
        self.tree.column("issue", width=90, anchor=tk.W)

        self.tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        tree_scroll = ttk.Scrollbar(left_frame, orient=tk.VERTICAL,
                                    command=self.tree.yview)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=tree_scroll.set)

        self.tree.tag_configure("error", background="#ffe6e6")
        self.tree.tag_configure("ok",    background="#e6ffe6")
        self.tree.bind("<<TreeviewSelect>>", self.on_word_select)

        # == Правая панель: графики (вкладки) ==
        right_frame = tk.Frame(main_pw, width=700)
        main_pw.add(right_frame, stretch="always")

        self.nb = ttk.Notebook(right_frame)
        self.nb.pack(fill=tk.BOTH, expand=True)

        # Вкладка 0: Сводка — ключевые метрики
        tab_summary = tk.Frame(self.nb)
        self.nb.add(tab_summary, text="📋 Сводка")
        self.txt_summary = tk.Text(tab_summary, font=("Arial", 12),
                                   wrap=tk.WORD, state=tk.DISABLED,
                                   padx=20, pady=20, bg="#fafafa")
        self.txt_summary.pack(fill=tk.BOTH, expand=True)

        # Вкладка 1: Осциллограмма + границы слогов
        tab_wave = tk.Frame(self.nb)
        self.nb.add(tab_wave, text="📈 Сигнал + слоги")

        self.fig_wave = Figure(figsize=(7, 3.5), dpi=90)
        self.ax_wave = self.fig_wave.add_subplot(111)
        self.canvas_wave = FigureCanvasTkAgg(self.fig_wave, tab_wave)
        self.canvas_wave.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._placeholder_wave()
        tb_wave = NavigationToolbar2Tk(self.canvas_wave, tab_wave)
        tb_wave.update()

        # Вкладка 2: Гласное пространство (F1/F2)
        tab_vowel = tk.Frame(self.nb)
        self.nb.add(tab_vowel, text="🎯 Гласные F1/F2")

        self.fig_f12 = Figure(figsize=(6.5, 5.5), dpi=90)
        self.ax_f12 = self.fig_f12.add_subplot(111)
        self.ax_f12.text(0.5, 0.5, "Запустите анализ, чтобы увидеть\nгласное пространство F1/F2",
                         ha='center', va='center', fontsize=13, color='gray',
                         transform=self.ax_f12.transAxes)
        self.canvas_f12 = FigureCanvasTkAgg(self.fig_f12, tab_vowel)
        self.canvas_f12.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        tb_f12 = NavigationToolbar2Tk(self.canvas_f12, tab_vowel)
        tb_f12.update()

        # Вкладка 3: Сводка редукции (длительность)
        tab_redux = tk.Frame(self.nb)
        self.nb.add(tab_redux, text="📊 Редукция по гласным")

        self.fig_redux = Figure(figsize=(6.5, 5.5), dpi=90)
        self.ax_redux = self.fig_redux.add_subplot(111)
        self.ax_redux.text(0.5, 0.5, "Запустите анализ, чтобы увидеть\nсводку редукции по гласным",
                           ha='center', va='center', fontsize=13, color='gray',
                           transform=self.ax_redux.transAxes)
        self.canvas_redux = FigureCanvasTkAgg(self.fig_redux, tab_redux)
        self.canvas_redux.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        tb_redux = NavigationToolbar2Tk(self.canvas_redux, tab_redux)
        tb_redux.update()

        # --- Нижняя панель ---
        bottom_frame = tk.Frame(self.root, pady=6, padx=12)
        bottom_frame.pack(fill=tk.X)

        self.lbl_status = tk.Label(bottom_frame,
            text="Готово к работе. Выберите файл.",
            font=("Arial", 10, "italic"))
        self.lbl_status.pack(side=tk.LEFT)

        self.progress = ttk.Progressbar(bottom_frame, orient=tk.HORIZONTAL,
                                        mode='indeterminate')
        self.progress.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(16, 0))

    # ═══════════════════════════════════════════════════════════
    #  ЗАГРУЗКА И АНАЛИЗ
    # ═══════════════════════════════════════════════════════════
    def _preload_models(self):
        """Load shared linguistic models; the selected ASR loads on demand."""
        try:
            # High-accuracy mode uses Whisper text with VOSK timing anchors,
            # so all shared models are prepared once in the background.
            self.models = pa.loadModels(verbose=False)
            self.root.after(0, lambda: self.lbl_status.config(
                text="Языковые модели загружены. Запишите или выберите аудио.",
                fg="#4CAF50"))
        except Exception as e:
            self.root.after(0, lambda: self.lbl_status.config(
                text=f"Ошибка загрузки моделей: {e}", fg="red"))

    def load_audio(self):
        filepath = filedialog.askopenfilename(
            title="Выберите аудио или видео с речью",
            filetypes=(("Аудио", "*.wav *.mp3 *.m4a *.flac *.ogg *.aac *.webm"),
                       ("Видео", "*.mp4 *.mov *.mkv *.webm"),
                       ("Все файлы", "*.*"))
        )
        if not filepath:
            return
        self.audio_path = filepath
        self.lbl_file.config(text=os.path.basename(filepath), fg="black")
        self.btn_analyze.config(state=tk.NORMAL)
        self.lbl_status.config(text="Файл загружен. Нажмите 'Анализировать'.",
                               fg="black")

    def _on_mode_changed(self):
        guided = self.mode_var.get() == "guided"
        self.cmb_phrase.config(state="readonly" if guided else tk.DISABLED)
        self.txt_reference.config(
            state=tk.NORMAL if guided else tk.DISABLED,
            background="white" if guided else "#eeeeee")
        if guided and not self.txt_reference.get("1.0", tk.END).strip():
            self.cmb_phrase.current(0)
            self._select_phrase()

    def _select_phrase(self, _event=None):
        phrase = self.cmb_phrase.get().strip()
        if not phrase:
            return
        self.txt_reference.config(state=tk.NORMAL)
        self.txt_reference.delete("1.0", tk.END)
        self.txt_reference.insert("1.0", phrase)

    def toggle_recording(self):
        """Start/stop microphone capture in the device's native sample rate."""
        if self.record_stream is not None:
            self._stop_recording()
            return
        try:
            import sounddevice as sd
            device = sd.query_devices(kind="input")
            self.record_sample_rate = int(device["default_samplerate"])
            self.record_chunks = []

            def callback(indata, _frames, _time, status):
                if status:
                    print(f"Микрофон: {status}")
                self.record_chunks.append(indata.copy())

            self.record_stream = sd.InputStream(
                samplerate=self.record_sample_rate,
                channels=1,
                dtype="float32",
                callback=callback)
            self.record_stream.start()
            self.btn_record.config(text="⏹ Остановить", bg="#f44336", fg="white")
            self.btn_load.config(state=tk.DISABLED)
            self.btn_analyze.config(state=tk.DISABLED)
            self.lbl_status.config(text="Идёт запись с микрофона…", fg="#d32f2f")
        except ImportError:
            messagebox.showerror(
                "Микрофон недоступен",
                "Установите модуль записи:\npython -m pip install sounddevice soundfile")
        except Exception as exc:
            self.record_stream = None
            messagebox.showerror("Ошибка микрофона", str(exc))

    def _stop_recording(self):
        stream, self.record_stream = self.record_stream, None
        try:
            stream.stop()
            stream.close()
            if not self.record_chunks:
                raise RuntimeError("Микрофон не вернул аудиоданные")
            import soundfile as sf
            samples = np.concatenate(self.record_chunks, axis=0)
            recordings_dir = os.path.join(os.path.dirname(pa.__file__), "recordings")
            os.makedirs(recordings_dir, exist_ok=True)
            filename = datetime.now().strftime("recording_%Y%m%d_%H%M%S.wav")
            path = os.path.join(recordings_dir, filename)
            sf.write(path, samples, self.record_sample_rate, subtype="PCM_16")
            self.audio_path = path
            self.lbl_file.config(text=filename, fg="black")
            self.btn_analyze.config(state=tk.NORMAL)
            self.lbl_status.config(
                text="Запись готова. Формат будет подготовлен автоматически.",
                fg="black")
        except Exception as exc:
            messagebox.showerror("Ошибка записи", str(exc))
        finally:
            self.record_chunks = []
            self.btn_record.config(text="🎙 Записать", bg="SystemButtonFace", fg="black")
            self.btn_load.config(state=tk.NORMAL)

    def on_close(self):
        if self.record_stream is not None:
            try:
                self.record_stream.stop()
                self.record_stream.close()
            except Exception:
                pass
        self.root.destroy()

    def start_analysis(self):
        if not self.audio_path:
            messagebox.showwarning("Нет аудио", "Выберите файл или запишите речь.")
            return
        if self.mode_var.get() == "guided":
            self.pending_reference = self.txt_reference.get("1.0", tk.END).strip()
            if not self.pending_reference:
                messagebox.showwarning(
                    "Нет фразы", "Выберите или введите фразу приложения.")
                return
        else:
            self.pending_reference = ""
        self.pending_asr = (
            "whisper" if self.asr_var.get().lower().startswith("whisper")
            else "vosk")
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.btn_analyze.config(state=tk.DISABLED)
        self.btn_load.config(state=tk.DISABLED)
        self.btn_record.config(state=tk.DISABLED)
        self.lbl_status.config(
            text="Подготовка аудио, распознавание и акустический анализ…",
            fg="blue")
        self.progress.start(15)
        threading.Thread(target=self.run_backend, daemon=True).start()

    def run_backend(self):
        try:
            result = pa.main(
                models=self.models,
                inputAudio=self.audio_path,
                referenceText=self.pending_reference,
                asrEngine=self.pending_asr)
            json_path = result["jsonPath"]
            self.analysis_audio_path = result["analysisAudio"]
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Загружаем сигнал для отрисовки осциллограмм
                self._load_audio_signal(self.analysis_audio_path)
                self.root.after(0, self.display_results, data)
            else:
                self.root.after(0, lambda: messagebox.showerror(
                    "Ошибка", "JSON файл не был создан!"))
        except Exception as e:
            error_text = str(e)
            self.root.after(0, lambda: messagebox.showerror(
                "Ошибка анализа", f"Произошла ошибка:\n{error_text}"))
            self.root.after(0, lambda: self.lbl_status.config(
                text="Ошибка анализа.", fg="red"))
        finally:
            self.root.after(0, self.progress.stop)
            self.root.after(0, lambda: self.btn_analyze.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.btn_load.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.btn_record.config(state=tk.NORMAL))

    def _load_audio_signal(self, path=None):
        """Читает аудио для визуализации."""
        path = path or self.audio_path
        try:
            import librosa
            self.full_signal, self.sr = librosa.load(
                path, sr=16000, mono=True, res_type="kaiser_fast")
        except Exception:
            # fallback: читаем через wave
            try:
                with wave.open(path, 'rb') as wf:
                    n = wf.getnframes()
                    raw = wf.readframes(n)
                    sr_wav = wf.getframerate()
                    self.sr = 16000
                    audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                    audio_np /= 32768.0
                    # ресемплинг до 16к
                    if sr_wav != 16000:
                        from scipy.signal import resample
                        n16k = int(len(audio_np) * 16000 / sr_wav)
                        audio_np = resample(audio_np, n16k)
                    self.full_signal = audio_np
            except Exception:
                self.full_signal = None

    def _placeholder_wave(self):
        """Начальная заглушка для вкладки осциллограммы."""
        self.ax_wave.text(0.5, 0.5,
            "Выберите слово в таблице слева,\nчтобы увидеть осциллограмму\nс границами слогов",
            ha='center', va='center', fontsize=13, color='gray',
            transform=self.ax_wave.transAxes)
        self.ax_wave.set_xticks([]); self.ax_wave.set_yticks([])
        self.canvas_wave.draw()

    # ═══════════════════════════════════════════════════════════
    #  ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ
    # ═══════════════════════════════════════════════════════════
    def display_results(self, data):
        self.analysis_data = data
        words = data.get("wordAnalysis", [])

        # Считаем метрики
        total_words = len(words)
        good_words = sum(1 for w in words if not w.get("hasIssue", False))
        multi_words = [w for w in words if w.get("syllableCount", 0) > 1]
        stress_match = sum(1 for w in multi_words
                          if w.get("expectedStressedIdx", -1) == w.get("actualStressedIdx", -2))
        redux_issues = sum(1 for w in words
                          if w.get("reductionInfo", {}).get("hasReductionIssue", False))

        wcmp = data.get("wordComparison", {})
        corr = len(wcmp.get("correct", []))
        total_ref = corr + len(wcmp.get("missed", [])) + len(wcmp.get("substituted", []))
        recognized_text = data.get("recognizedText", "").strip()
        rec_diag = data.get("recognitionDiagnostics", {})
        mean_conf = rec_diag.get("meanWordConfidence")
        low_conf = rec_diag.get("lowConfidenceWordCount", 0)
        recognition_engine = data.get("recognitionEngine", "vosk")
        recognition_device = data.get("recognitionDevice", "cpu")
        v2_diag = data.get("acousticNucleiV2", {})
        v2_total = v2_diag.get("count", 0)
        v2_assigned = v2_diag.get("assignedToRecognizedWords", 0)
        v2_mean_conf = v2_diag.get("meanCandidateConfidence")
        v2_count_matches = sum(
            1 for w in words if not w.get("v2CountMismatch", True))

        # --- Заполняем сводку ---
        self.txt_summary.config(state=tk.NORMAL)
        self.txt_summary.delete("1.0", tk.END)
        lines = [
            "═══════════════════════════════════",
            "   СВОДКА АНАЛИЗА",
            "═══════════════════════════════════",
            "",
            f"📁 Файл: {os.path.basename(self.audio_path)}",
            f"⏱  Длительность: {data.get('durationSec', 0):.1f} с",
            f"🔤 Слов распознано: {total_words}",
            f"🧠 Распознаватель: {recognition_engine} ({recognition_device})",
            "",
            "── Автоматическая расшифровка ──",
            recognized_text or "(речь не распознана)",
            "",
            "── Диагностические сигналы (не общий балл) ──",
            (f"✅ Без срабатывания эвристик: {good_words}/{total_words} "
             f"({good_words/total_words*100:.0f}%)" if total_words else
             "Распознанных слов для анализа нет."),
            f"⚠ Требуют проверки: {total_words - good_words}/{total_words}",
            "",
            "── Ударение: акустика ↔ словарь распознанного слова ──",
            f"🎯 Совпадений: {stress_match}/{len(multi_words)}",
        ]
        if len(multi_words) > 0:
            lines.append(
                f"   ({stress_match/len(multi_words)*100:.0f}%; это согласованность, не ground truth)")
        else:
            lines.append("   (нет многосложных слов)")

        if mean_conf is not None:
            lines.extend([
                "",
                "── Надёжность распознавания ──",
                f"Средняя уверенность распознавателя: {mean_conf:.2f}",
                f"Слов с низкой уверенностью (<0.70): {low_conf}/{total_words}",
                "Уверенность модели не равна измеренной точности распознавания.",
            ])

        if v2_diag:
            lines.extend([
                "",
                "── Акустические ядра V2 (без текста) ──",
                f"Найдено в записи: {v2_total}; привязано к словам: {v2_assigned}",
                f"Совпадение числа V2-ядер с распознанным текстом: "
                f"{v2_count_matches}/{total_words}",
                "V2 ищет ядра только по аудио. Текст здесь показан лишь для "
                "сравнения и не управляет детектором.",
            ])
            if v2_mean_conf is not None:
                lines.append(
                    f"Средняя уверенность кандидатов: {v2_mean_conf:.2f} "
                    "(это не измеренная точность).")

        lines.append("")
        lines.append("── Редукция гласных ──")
        lines.append(f"🔄 Слов с подозрением на недостаточную редукцию: {redux_issues}/{total_words}")

        if total_ref > 0:
            lines.append("")
            lines.append("── Сравнение с эталонным текстом ──")
            acc = corr / total_ref * 100
            lines.append(f"📝 Точность слов: {acc:.1f}% ({corr}/{total_ref})")
            missed = len(wcmp.get("missed", []))
            sub = len(wcmp.get("substituted", []))
            if missed: lines.append(f"   Пропущено: {missed}")
            if sub: lines.append(f"   Замен: {sub}")
        else:
            lines.append("")
            lines.append("── Свободная речь ──")
            lines.append(
                "Текст введён не был: приложение анализирует автоматическую "
                "расшифровку, темп, паузы и акустические признаки.")
            lines.append(
                "Без заданной фразы нельзя определить, какое слово пользователь "
                "намеревался произнести; низкоуверенные слова нужно трактовать осторожно.")

        self.txt_summary.insert("1.0", "\n".join(lines))
        self.txt_summary.config(state=tk.DISABLED)

        # --- Заполняем таблицу ---
        for w in words:
            word_text = w.get("word", "")
            exp_n = w.get("textNucleusCount", w.get("syllableCount", 0))
            v1_n = w.get("detectedNucleiV1", w.get("detectedNuclei", 0))
            v2_n = w.get("detectedNucleiV2", "—")
            syl_str = f"{v2_n} / {v1_n} / {exp_n}"
            exp_s = w.get("expectedStressedIdx", -1)
            act_s = w.get("actualStressedIdx", -1)
            if w.get("countMismatch"):
                stress_str = "—"
            elif exp_s == act_s:
                stress_str = "✓ Совпадает"
            else:
                stress_str = f"✗ Слов.{exp_s+1}→Акуст.{act_s+1}"
            red = w.get("reductionInfo", {})
            red_str = "⚠" if red.get("hasReductionIssue") else "✓"
            has = w.get("hasIssue", False)
            issue_str = "⚠ Проверить" if has else "✓ Нет сигнала"
            tag = "error" if has else "ok"
            self.tree.insert("", tk.END,
                values=(word_text, syl_str, stress_str, red_str, issue_str),
                tags=(tag,))

        if total_ref > 0:
            acc = corr / total_ref * 100
            self.lbl_status.config(
                text=f"Анализ завершён! Точность текста: {acc:.1f}%",
                fg="#4CAF50")
        else:
            self.lbl_status.config(text="Анализ завершён!", fg="#4CAF50")

        # Рисуем гласное пространство и сводку редукции
        self._draw_vowel_space()
        self._draw_reduction_summary()

    # ═══════════════════════════════════════════════════════════
    #  ВИЗУАЛИЗАЦИЯ: осциллограмма слова
    # ═══════════════════════════════════════════════════════════
    def on_word_select(self, event):
        """При клике на слово — осциллограмма + границы слогов."""
        sel = self.tree.selection()
        if not sel or self.analysis_data is None:
            return
        idx = self.tree.index(sel[0])
        words = self.analysis_data.get("wordAnalysis", [])
        if idx >= len(words):
            return
        w = words[idx]
        self._draw_word_waveform(w)

    def _draw_word_waveform(self, w):
        """Рисует осциллограмму слова с границами слогов."""
        self.ax_wave.clear()

        if self.full_signal is None:
            self.ax_wave.text(0.5, 0.5, "Аудио не загружено",
                              ha='center', va='center', transform=self.ax_wave.transAxes)
            self.canvas_wave.draw()
            return

        start_t = w.get("start", 0)
        end_t   = w.get("end", 0)
        v2_nuclei = w.get("v2Nuclei", [])
        plot_start_t = start_t
        # ASR can trim the final vowel slightly. Extend only far enough to
        # display a V2 point that was assigned within the 80 ms tolerance.
        if v2_nuclei:
            first_v2 = min(point.get("timeSec", start_t) for point in v2_nuclei)
            last_v2 = max(point.get("timeSec", end_t) for point in v2_nuclei)
            plot_start_t = max(0.0, min(start_t, first_v2 - 0.025))
            end_t = max(end_t, last_v2 + 0.025)
        s = int(plot_start_t * self.sr)
        e = int(end_t * self.sr)
        s = max(0, s); e = min(len(self.full_signal), e)
        if e - s < 10:
            self.ax_wave.text(0.5, 0.5, "Слишком короткий сегмент",
                              ha='center', va='center', transform=self.ax_wave.transAxes)
            self.canvas_wave.draw()
            return

        seg = self.full_signal[s:e]
        t_axis = (plot_start_t - start_t) + np.arange(len(seg)) / self.sr

        self.ax_wave.plot(t_axis, seg, color='#3366cc', linewidth=0.6)

        # V2 nuclei are independent acoustic observations. They are drawn as
        # solid purple lines; dashed gray/red lines below remain V1 syllable
        # boundaries used by the legacy stress/reduction pipeline.
        v2_label_drawn = False
        for point in v2_nuclei:
            local_time = point.get(
                "timeInWordSec", point.get("timeSec", start_t) - start_t)
            if t_axis[0] <= local_time <= t_axis[-1]:
                self.ax_wave.axvline(
                    local_time, color='#7b1fa2', linewidth=1.6, alpha=0.9,
                    label="ядро V2" if not v2_label_drawn else None)
                v2_label_drawn = True

        # Границы слогов
        syls = w.get("syllableAnalysis", [])
        stressed = w.get("expectedStressedIdx", -1)
        colors_v = ['#e91e63', '#2196f3', '#4caf50', '#ff9800', '#9c27b0',
                    '#00bcd4', '#795548', '#607d8b']

        for i, syl in enumerate(syls):
            ss = syl.get("startSec", 0)
            se = syl.get("endSec", 0)
            syl_text = syl.get("syllable", "")
            mid = (ss + se) / 2
            is_str = (i == stressed)

            # прямоугольник фона для ударного слога
            if is_str:
                self.ax_wave.axvspan(ss, se, alpha=0.15, color='#ff5722')
            # вертикальные линии границ
            self.ax_wave.axvline(ss, color='#555555' if not is_str else '#d32f2f',
                                 linestyle='--', linewidth=0.8, alpha=0.7)
            # метка слога
            color = '#d32f2f' if is_str else '#333333'
            weight = 'bold' if is_str else 'normal'
            marker = f"«{syl_text}»" + (" ´" if is_str else "")
            self.ax_wave.text(mid, seg.max() * 0.92, marker,
                              ha='center', va='bottom', fontsize=9,
                              color=color, fontweight=weight)

        # последняя граница
        if syls:
            self.ax_wave.axvline(syls[-1].get("endSec", 0), color='#555555',
                                 linestyle='--', linewidth=0.8, alpha=0.7)

        # если есть вокалическое ядро — подсветка
        for syl in syls:
            ac = syl.get("acoustics", {})
            core = ac.get("core")
            if core:
                cs = core.get("startInSyl", 0)
                ce = core.get("endInSyl", 0)
                if ce > cs:
                    self.ax_wave.axvspan(cs, ce, alpha=0.08, color='#ffeb3b')

        word_text = w.get("word", "")
        exp_s = w.get("expectedStressedIdx", -1)
        act_s = w.get("actualStressedIdx", -1)
        v1_count = w.get("detectedNucleiV1", w.get("detectedNuclei", "?"))
        v2_count = w.get("detectedNucleiV2", "?")
        text_count = w.get("textNucleusCount", len(syls))
        title = (f"«{word_text}»  |  ядра V2 / V1 / текст: "
                 f"{v2_count} / {v1_count} / {text_count}")
        if exp_s >= 0:
            title += f"  |  ударение: словарь→слог{exp_s+1}"
        if not w.get("countMismatch"):
            title += f"  акустика→слог{act_s+1}"
        if v2_label_drawn:
            self.ax_wave.legend(loc="upper right", fontsize=8)

        self.ax_wave.set_title(title, fontsize=11, fontweight='bold')
        self.ax_wave.set_xlabel("Время (с)")
        self.ax_wave.set_ylabel("Амплитуда")
        # оставляем небольшой отступ справа
        padding = max(0.005, (t_axis[-1] - t_axis[0]) * 0.02)
        self.ax_wave.set_xlim(t_axis[0] - padding, t_axis[-1] + padding)
        self.ax_wave.set_ylim(seg.min() * 1.2, seg.max() * 1.2)

        self.fig_wave.tight_layout()
        self.canvas_wave.draw()

    # ═══════════════════════════════════════════════════════════
    #  ВИЗУАЛИЗАЦИЯ: гласное пространство F1/F2
    # ═══════════════════════════════════════════════════════════
    def _draw_vowel_space(self):
        """Рисует F1/F2 scatter для всех гласных в записи."""
        self.ax_f12.clear()
        if self.analysis_data is None:
            self.canvas_f12.draw()
            return

        words = self.analysis_data.get("wordAnalysis", [])
        stressed_pts = []    # (F1, F2, label)
        unstr_pts    = []

        for w in words:
            if w.get("syllableCount", 0) < 2:
                continue
            st_i = w.get("expectedStressedIdx", -1)
            for i, syl in enumerate(w.get("syllableAnalysis", [])):
                ac = syl.get("acoustics", {})
                # Используем core если есть, иначе полный слог
                src = ac.get("core") if ac.get("core") else ac
                fmts = src.get("formants", [])
                if len(fmts) < 2:
                    continue
                f1, f2 = fmts[0], fmts[1]
                if f1 < 100 or f2 < 500:
                    continue
                syl_text = syl.get("syllable", "")
                # главный гласный слога
                vowel = None
                for ch in syl_text.lower():
                    if ch in "аоеияёэюуы":
                        vowel = ch
                        break
                label = vowel if vowel else syl_text

                if i == st_i:
                    stressed_pts.append((f1, f2, label))
                else:
                    unstr_pts.append((f1, f2, label))

        # Схематические области гласных (приблизительно для русского)
        # Рисуем только подписи, без жёстких эллипсов
        vowel_labels = {
            'а': (750, 1300), 'о': (530, 900),  'у': (370, 650),
            'е': (450, 1900), 'и': (300, 2200), 'ы': (380, 1400),
        }
        schwa_f1, schwa_f2 = 500, 1500

        # Нейтральный центр [ə]
        self.ax_f12.scatter([schwa_f2], [schwa_f1], marker='x', color='gray',
                            s=120, linewidths=2, zorder=10)
        self.ax_f12.annotate('[ə]', (schwa_f2 + 30, schwa_f1 + 30),
                             fontsize=10, color='gray')

        # Ударные
        if stressed_pts:
            sf1 = [p[0] for p in stressed_pts]
            sf2 = [p[1] for p in stressed_pts]
            self.ax_f12.scatter(sf2, sf1, c='#d32f2f', marker='o',
                                s=36, alpha=0.7, edgecolors='#880000',
                                linewidth=0.5, label='Ударные', zorder=5)

        # Безударные
        if unstr_pts:
            uf1 = [p[0] for p in unstr_pts]
            uf2 = [p[1] for p in unstr_pts]
            self.ax_f12.scatter(uf2, uf1, c='#2196f3', marker='o',
                                s=22, alpha=0.5, edgecolors='#003388',
                                linewidth=0.3, label='Безударные', zorder=4)

        # Подписи областей
        for v, (f1, f2) in vowel_labels.items():
            self.ax_f12.annotate(f'[{v}]', (f2 + 40, f1 - 20),
                                 fontsize=11, color='#555555',
                                 fontstyle='italic')

        self.ax_f12.set_xlabel("F2 (Гц)", fontsize=11)
        self.ax_f12.set_ylabel("F1 (Гц)", fontsize=11)
        self.ax_f12.set_title("Гласное пространство (F1/F2) — "
                              "ударные vs безударные", fontsize=12)
        self.ax_f12.legend(loc='upper left', fontsize=9)
        self.ax_f12.invert_xaxis()
        self.ax_f12.invert_yaxis()
        self.ax_f12.grid(True, alpha=0.3)
        self.fig_f12.tight_layout()
        self.canvas_f12.draw()

    # ═══════════════════════════════════════════════════════════
    #  ВИЗУАЛИЗАЦИЯ: сводка редукции по гласным
    # ═══════════════════════════════════════════════════════════
    def _draw_reduction_summary(self):
        """Группирует слоги по гласному и позиции, строит bar chart."""
        self.ax_redux.clear()
        if self.analysis_data is None:
            self.canvas_redux.draw()
            return

        from collections import defaultdict
        words = self.analysis_data.get("wordAnalysis", [])
        # Собираем: vowel -> position -> list of durations
        pool = defaultdict(lambda: defaultdict(list))
        for w in words:
            if w.get("syllableCount", 0) < 2:
                continue
            st_i = w.get("expectedStressedIdx", -1)
            for i, syl in enumerate(w.get("syllableAnalysis", [])):
                ac = syl.get("acoustics", {})
                src = ac.get("core") if ac.get("core") else ac
                dur = src.get("duration", 0)
                if dur < 0.025:
                    continue
                syl_text = syl.get("syllable", "")
                vowel = None
                for ch in syl_text.lower():
                    if ch in "аоеия":
                        vowel = ch
                        break
                if vowel is None:
                    continue
                if i == st_i:
                    pool[vowel]["ударный"].append(dur)
                elif i == st_i - 1:
                    pool[vowel]["предуд."].append(dur)
                elif i == st_i + 1:
                    pool[vowel]["зауд."].append(dur)
                else:
                    pool[vowel]["проч."].append(dur)

        # Фильтруем: только гласные с ≥3 образцами в каждой позиции
        vowels = sorted(pool.keys())
        pos_order = ["ударный", "предуд.", "зауд.", "проч."]
        pos_colors = {"ударный": "#d32f2f", "предуд.": "#ff9800",
                      "зауд.": "#2196f3", "проч.": "#9e9e9e"}

        x_labels = []
        bar_data = {p: [] for p in pos_order}
        for v in vowels:
            d = pool[v]
            if d.get("ударный") and len(d["ударный"]) >= 2:
                x_labels.append(v)
                for p in pos_order:
                    vals = d.get(p, [])
                    bar_data[p].append(np.mean(vals) if vals else 0)

        if not x_labels:
            self.ax_redux.text(0.5, 0.5, "Недостаточно данных",
                               ha='center', va='center',
                               transform=self.ax_redux.transAxes)
            self.canvas_redux.draw()
            return

        x = np.arange(len(x_labels))
        width = 0.18
        for i, p in enumerate(pos_order):
            offset = (i - 1.5) * width
            vals = bar_data[p]
            self.ax_redux.bar(x + offset, vals, width, label=p,
                              color=pos_colors[p], alpha=0.85,
                              edgecolor='white', linewidth=0.5)

        self.ax_redux.set_xticks(x)
        self.ax_redux.set_xticklabels([f"«{v}»" for v in x_labels],
                                       fontsize=12)
        self.ax_redux.set_ylabel("Средняя длительность (с)", fontsize=11)
        self.ax_redux.set_title("Длительность вокалического ядра "
                                "по позициям", fontsize=12)
        self.ax_redux.legend(fontsize=8, ncol=4)
        self.ax_redux.grid(axis='y', alpha=0.3)
        self.fig_redux.tight_layout()
        self.canvas_redux.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = PronunciationGUI(root)
    root.mainloop()
