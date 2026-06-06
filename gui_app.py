import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import json
import threading
import numpy as np
import wave

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

        self.create_widgets()

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

        self.btn_load = tk.Button(top_frame, text="📁 Выбрать аудио (.wav)",
                                  font=("Arial", 11), command=self.load_audio)
        self.btn_load.pack(side=tk.LEFT, padx=4)

        self.lbl_file = tk.Label(top_frame, text="Файл не выбран",
                                 font=("Arial", 11), fg="gray")
        self.lbl_file.pack(side=tk.LEFT, padx=12)

        self.btn_analyze = tk.Button(top_frame, text="▶ Анализировать",
                                     font=("Arial", 11, "bold"),
                                     bg="#4CAF50", fg="white",
                                     command=self.start_analysis,
                                     state=tk.DISABLED)
        self.btn_analyze.pack(side=tk.RIGHT, padx=4)

        # --- Текст ---
        text_frame = tk.Frame(self.root, pady=3, padx=12)
        text_frame.pack(fill=tk.X)

        tk.Label(text_frame, text="Эталонный текст:",
                font=("Arial", 10, "bold")).pack(anchor=tk.W)

        self.txt_reference = tk.Text(text_frame, height=3,
                                     font=("Arial", 11), wrap=tk.WORD)
        self.txt_reference.pack(fill=tk.X, pady=3)
        self.txt_reference.insert(tk.END,
            "(необязательно) Эталонный текст для сравнения. Если оставить пустым — анализ только по акустике.")

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
        self.tree.heading("syllables", text="Слоги (факт/норма)")
        self.tree.heading("stress", text="Ударение (акуст. vs словарь)")
        self.tree.heading("reduction", text="Редукция")
        self.tree.heading("issue", text="Статус")

        self.tree.column("word", width=100, anchor=tk.W)
        self.tree.column("syllables", width=110, anchor=tk.CENTER)
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

        # Вкладка 1: Осциллограмма + границы слогов
        tab_wave = tk.Frame(self.nb)
        self.nb.add(tab_wave, text="📈 Осциллограмма + слоги")

        self.fig_wave = Figure(figsize=(7, 3.5), dpi=90)
        self.ax_wave = self.fig_wave.add_subplot(111)
        self.canvas_wave = FigureCanvasTkAgg(self.fig_wave, tab_wave)
        self.canvas_wave.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        tb_wave = NavigationToolbar2Tk(self.canvas_wave, tab_wave)
        tb_wave.update()

        # Вкладка 2: Гласное пространство (F1/F2)
        tab_vowel = tk.Frame(self.nb)
        self.nb.add(tab_vowel, text="🎯 Гласные F1/F2")

        self.fig_f12 = Figure(figsize=(6.5, 5.5), dpi=90)
        self.ax_f12 = self.fig_f12.add_subplot(111)
        self.canvas_f12 = FigureCanvasTkAgg(self.fig_f12, tab_vowel)
        self.canvas_f12.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        tb_f12 = NavigationToolbar2Tk(self.canvas_f12, tab_vowel)
        tb_f12.update()

        # Вкладка 3: Сводка редукции (длительность)
        tab_redux = tk.Frame(self.nb)
        self.nb.add(tab_redux, text="📊 Редукция по гласным")

        self.fig_redux = Figure(figsize=(6.5, 5.5), dpi=90)
        self.ax_redux = self.fig_redux.add_subplot(111)
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
        """Фоновая загрузка VOSK + pymorphy2 + ruaccent при старте."""
        try:
            self.models = pa.loadModels(verbose=False)
            self.root.after(0, lambda: self.lbl_status.config(
                text="Модели загружены. Выберите аудиофайл.", fg="#4CAF50"))
        except Exception as e:
            self.root.after(0, lambda: self.lbl_status.config(
                text=f"Ошибка загрузки моделей: {e}", fg="red"))

    def load_audio(self):
        filepath = filedialog.askopenfilename(
            title="Выберите WAV файл",
            filetypes=(("WAV files", "*.wav"), ("All files", "*.*"))
        )
        if not filepath:
            return
        self.audio_path = filepath
        self.lbl_file.config(text=os.path.basename(filepath), fg="black")
        self.btn_analyze.config(state=tk.NORMAL)
        self.lbl_status.config(text="Файл загружен. Нажмите 'Анализировать'.",
                               fg="black")

        # Автопоиск текста
        base_name = os.path.basename(filepath)
        txt_name = base_name.replace(".wav", ".txt")
        audio_dir = os.path.dirname(filepath)
        parent_dir = os.path.dirname(audio_dir)

        possible = [
            os.path.join(audio_dir, txt_name),
            os.path.join(parent_dir, "text", txt_name),
            os.path.join(os.path.dirname(pa.__file__), "text", txt_name),
        ]
        self.txt_reference.delete("1.0", tk.END)
        for p in possible:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    self.txt_reference.insert(tk.END, f.read().strip())
                return
        self.txt_reference.insert(tk.END, "")

    def start_analysis(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.btn_analyze.config(state=tk.DISABLED)
        self.btn_load.config(state=tk.DISABLED)
        self.lbl_status.config(
            text="Идёт анализ (VOSK + поиск ядер + вокалическое ядро)...",
            fg="blue")
        self.progress.start(15)
        threading.Thread(target=self.run_backend, daemon=True).start()

    def run_backend(self):
        temp_txt = None
        try:
            pa.audioFile = self.audio_path
            user_text = self.txt_reference.get("1.0", tk.END).strip()
            is_hint = user_text.startswith("(необязательно)")
            if user_text and not is_hint:
                temp_txt = os.path.join(os.path.dirname(pa.__file__),
                                        "temp_reference.txt")
                with open(temp_txt, "w", encoding="utf-8") as f:
                    f.write(user_text)
                pa.textFile = temp_txt
            else:
                # Без эталонного текста — только акустический анализ
                pa.textFile = os.path.join(os.path.dirname(pa.__file__),
                                           "_no_reference_.txt")
            pa.main(models=self.models)

            base = os.path.basename(self.audio_path).replace(".wav",
                                            "_syllable_analysis.json")
            json_path = os.path.join(os.path.dirname(pa.__file__),
                                     "analysis", base)
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Загружаем сигнал для отрисовки осциллограмм
                self._load_audio_signal()
                self.root.after(0, self.display_results, data)
            else:
                self.root.after(0, lambda: messagebox.showerror(
                    "Ошибка", "JSON файл не был создан!"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror(
                "Ошибка анализа", f"Произошла ошибка:\n{str(e)}"))
            self.root.after(0, lambda: self.lbl_status.config(
                text="Ошибка анализа.", fg="red"))
        finally:
            if temp_txt and os.path.exists(temp_txt):
                os.remove(temp_txt)
            self.root.after(0, self.progress.stop)
            self.root.after(0, lambda: self.btn_analyze.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.btn_load.config(state=tk.NORMAL))

    def _load_audio_signal(self):
        """Читает аудио для визуализации."""
        try:
            import librosa
            self.full_signal, self.sr = librosa.load(
                self.audio_path, sr=16000, mono=True, res_type="kaiser_fast")
        except Exception:
            # fallback: читаем через wave
            try:
                with wave.open(self.audio_path, 'rb') as wf:
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

    # ═══════════════════════════════════════════════════════════
    #  ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ
    # ═══════════════════════════════════════════════════════════
    def display_results(self, data):
        self.analysis_data = data
        words = data.get("wordAnalysis", [])
        for w in words:
            word_text = w.get("word", "")

            exp_n = w.get("syllableCount", 0)
            act_n = w.get("detectedNuclei", 0)
            syl_str = f"{act_n} / {exp_n}"

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
            issue_str = "⚠ Внимание" if has else "✓ Отлично"
            tag = "error" if has else "ok"

            self.tree.insert("", tk.END,
                values=(word_text, syl_str, stress_str, red_str, issue_str),
                tags=(tag,))

        # метрики
        wcmp = data.get("wordComparison", {})
        corr = len(wcmp.get("correct", []))
        total = corr + len(wcmp.get("missed", [])) + len(
                     wcmp.get("substituted", []))
        if total > 0:
            acc = corr / total * 100
            self.lbl_status.config(
                text=f"Анализ завершён! Точность текста: {acc:.1f}%",
                fg="#4CAF50")
        else:
            self.lbl_status.config(text="Анализ завершён!", fg="#4CAF50")

        # Рисуем гласное пространство
        self._draw_vowel_space()

        # Рисуем сводку редукции
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
        s = int(start_t * self.sr)
        e = int(end_t * self.sr)
        s = max(0, s); e = min(len(self.full_signal), e)
        if e - s < 10:
            self.ax_wave.text(0.5, 0.5, "Слишком короткий сегмент",
                              ha='center', va='center', transform=self.ax_wave.transAxes)
            self.canvas_wave.draw()
            return

        seg = self.full_signal[s:e]
        t_axis = np.linspace(0, len(seg) / self.sr, len(seg))

        self.ax_wave.plot(t_axis, seg, color='#3366cc', linewidth=0.6)

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
        title = f"«{word_text}»  |  слогов: {len(syls)}"
        if exp_s >= 0:
            title += f"  |  ударение: словарь→слог{exp_s+1}"
        if not w.get("countMismatch"):
            title += f"  акустика→слог{act_s+1}"
        if w.get("countMismatch"):
            nuclei = w.get("detectedNuclei", "?")
            title += f"  [ядер={nuclei}]"

        self.ax_wave.set_title(title, fontsize=11, fontweight='bold')
        self.ax_wave.set_xlabel("Время (с)")
        self.ax_wave.set_ylabel("Амплитуда")
        # оставляем небольшой отступ справа
        self.ax_wave.set_xlim(0, t_axis[-1] * 1.02)
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
