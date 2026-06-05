import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import json
import threading

try:
    import pronunciationAnalyzer as pa
except ImportError:
    messagebox.showerror("Ошибка", "Не найден файл pronunciationAnalyzer.py! Убедитесь, что он в той же папке.")

class PronunciationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализатор произношения (Система оценки речи)")
        self.root.geometry("1000x750")
        
        # Настройка современных стилей
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Treeview.Heading", font=("Arial", 10, "bold"), background="#e0e0e0")
        style.configure("Treeview", font=("Arial", 10), rowheight=28)
        
        self.audio_path = None
        self.create_widgets()

    def create_widgets(self):
        # --- Верхняя панель (Загрузка аудио) ---
        top_frame = tk.Frame(self.root, pady=10, padx=15)
        top_frame.pack(fill=tk.X)

        self.btn_load = tk.Button(top_frame, text="📁 Выбрать аудио (.wav)", font=("Arial", 11), command=self.load_audio)
        self.btn_load.pack(side=tk.LEFT, padx=5)

        self.lbl_file = tk.Label(top_frame, text="Файл не выбран", font=("Arial", 11), fg="gray")
        self.lbl_file.pack(side=tk.LEFT, padx=15)

        self.btn_analyze = tk.Button(top_frame, text="▶ Анализировать", font=("Arial", 11, "bold"), 
                                     bg="#4CAF50", fg="white", command=self.start_analysis, state=tk.DISABLED)
        self.btn_analyze.pack(side=tk.RIGHT, padx=5)

        # --- НОВОЕ: Текстовая панель (Для эталонного текста) ---
        text_frame = tk.Frame(self.root, pady=5, padx=15)
        text_frame.pack(fill=tk.X)
        
        lbl_txt = tk.Label(text_frame, text="Эталонный текст (Вставьте текст, который вы читаете):", font=("Arial", 10, "bold"))
        lbl_txt.pack(anchor=tk.W)
        
        self.txt_reference = tk.Text(text_frame, height=4, font=("Arial", 11), wrap=tk.WORD)
        self.txt_reference.pack(fill=tk.X, pady=5)
        self.txt_reference.insert(tk.END, "Загрузите аудио, чтобы авто-найти текст, или просто вставьте его сюда.")

        # --- Центральная панель (Результаты) ---
        mid_frame = tk.Frame(self.root, pady=5, padx=15)
        mid_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("word", "syllables", "stress_status", "reduction", "issue")
        self.tree = ttk.Treeview(mid_frame, columns=columns, show="headings")
        self.tree.heading("word", text="Слово")
        self.tree.heading("syllables", text="Слоги (Факт / Норма)")
        self.tree.heading("stress_status", text="Ударение (Акустика vs Словарь)")
        self.tree.heading("reduction", text="Редукция гласных")
        self.tree.heading("issue", text="Статус")
        
        self.tree.column("word", width=160, anchor=tk.W)
        self.tree.column("syllables", width=160, anchor=tk.CENTER)
        self.tree.column("stress_status", width=240, anchor=tk.CENTER)
        self.tree.column("reduction", width=140, anchor=tk.CENTER)
        self.tree.column("issue", width=180, anchor=tk.W)
        
        self.tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        scrollbar = ttk.Scrollbar(mid_frame, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.tag_configure("error", background="#ffe6e6") # Светло-красный
        self.tree.tag_configure("ok", background="#e6ffe6")    # Светло-зеленый

        # --- Нижняя панель (Статус и Прогресс-бар) ---
        bottom_frame = tk.Frame(self.root, pady=10, padx=15)
        bottom_frame.pack(fill=tk.X)
        
        self.lbl_status = tk.Label(bottom_frame, text="Готово к работе. Выберите файл.", font=("Arial", 10, "italic"))
        self.lbl_status.pack(side=tk.LEFT)

        self.progress = ttk.Progressbar(bottom_frame, orient=tk.HORIZONTAL, mode='indeterminate')
        self.progress.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(20, 0))

    def load_audio(self):
        filepath = filedialog.askopenfilename(
            title="Выберите WAV файл",
            filetypes=(("WAV files", "*.wav"), ("All files", "*.*"))
        )
        if filepath:
            self.audio_path = filepath
            self.lbl_file.config(text=os.path.basename(filepath), fg="black")
            self.btn_analyze.config(state=tk.NORMAL)
            self.lbl_status.config(text="Файл загружен. Нажмите 'Анализировать'.", fg="black")
            
            # Пытаемся автоматически найти .txt файл
            base_name = os.path.basename(filepath)
            txt_name = base_name.replace(".wav", ".txt")
            audio_dir = os.path.dirname(filepath)
            parent_dir = os.path.dirname(audio_dir)
            
            possible_txt_paths = [
                os.path.join(audio_dir, txt_name),
                os.path.join(parent_dir, "text", txt_name),
                os.path.join(os.path.dirname(pa.__file__), "text", txt_name)
            ]
            
            self.txt_reference.delete("1.0", tk.END)
            text_found = False
            for p in possible_txt_paths:
                if os.path.exists(p):
                    with open(p, "r", encoding="utf-8") as f:
                        self.txt_reference.insert(tk.END, f.read().strip())
                    text_found = True
                    break
            
            if not text_found:
                self.txt_reference.insert(tk.END, "") # Оставляем пустым, чтобы пользователь сам вставил

    def start_analysis(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        self.btn_analyze.config(state=tk.DISABLED)
        self.btn_load.config(state=tk.DISABLED)
        self.lbl_status.config(text="Идет анализ (извлечение MFCC, VOSK, поиск ядер)...", fg="blue")
        self.progress.start(15) 
        
        threading.Thread(target=self.run_backend, daemon=True).start()

    def run_backend(self):
        temp_txt_path = "temp_reference.txt"
        try:
            pa.audioFile = self.audio_path
            
            # Создаем временный файл с текстом из текстового поля GUI
            user_text = self.txt_reference.get("1.0", tk.END).strip()
            with open(temp_txt_path, "w", encoding="utf-8") as f:
                f.write(user_text)
            
            pa.textFile = temp_txt_path
            
            # Запускаем скрипт
            pa.main()

            json_path = self.audio_path.replace(".wav", "_syllable_analysis.json")
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.root.after(0, self.display_results, data)
            else:
                self.root.after(0, lambda: messagebox.showerror("Ошибка", "JSON файл не был создан!"))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Ошибка анализа", f"Произошла ошибка:\n{str(e)}"))
            self.root.after(0, lambda: self.lbl_status.config(text="Ошибка анализа.", fg="red"))
        finally:
            # Удаляем временный файл
            if os.path.exists(temp_txt_path):
                os.remove(temp_txt_path)
                
            self.root.after(0, self.progress.stop)
            self.root.after(0, lambda: self.btn_analyze.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.btn_load.config(state=tk.NORMAL))

    def display_results(self, data):
        words = data.get("wordAnalysis", [])
        for w in words:
            word_text = w.get("word", "")
            
            syl_count_expected = w.get("syllableCount", 0)
            syl_count_actual = w.get("detectedNuclei", 0)
            syl_str = f"{syl_count_actual} / {syl_count_expected}"
            
            exp_stress = w.get("expectedStressedIdx", -1)
            act_stress = w.get("actualStressedIdx", -1)
            
            if w.get("countMismatch"):
                stress_str = "—" 
            else:
                stress_str = "✓ Совпадает" if exp_stress == act_stress else f"✗ Ошибка (Норма: {exp_stress+1}, Факт: {act_stress+1})"
            
            red_info = w.get("reductionInfo", {})
            has_red_issue = red_info.get("hasReductionIssue", False)
            red_str = "⚠ Проблема" if has_red_issue else "✓ ОК"
            
            has_issue = w.get("hasIssue", False)
            issue_str = "⚠ Требует внимания" if has_issue else "✓ Отлично"
            
            tag = "error" if has_issue else "ok"

            self.tree.insert("", tk.END, values=(word_text, syl_str, stress_str, red_str, issue_str), tags=(tag,))
            
        word_cmp = data.get("wordComparison", {})
        correct = len(word_cmp.get("correct", []))
        total_ref = correct + len(word_cmp.get("missed", [])) + len(word_cmp.get("substituted", []))
        
        if total_ref > 0:
            acc = (correct / total_ref * 100)
            status_text = f"Анализ завершен! Точность текста: {acc:.1f}%"
        else:
            status_text = "Анализ завершен! (Режим без эталонного текста)"
            
        self.lbl_status.config(text=status_text, fg="#4CAF50")

if __name__ == "__main__":
    root = tk.Tk()
    app = PronunciationGUI(root)
    root.mainloop()