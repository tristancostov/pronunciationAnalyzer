import sys, os
print(sys.executable)

# 复现脚本的导入顺序
import wave, json, re, difflib
import numpy as np
import librosa
from scipy.signal import find_peaks, butter, sosfiltfilt

from ruaccent import RUAccent     # 这一行如果让你的脚本崩,就是它
accentizer = RUAccent()
accentizer.load(omograph_model_size="turbo", use_dictionary=True)
print("ruaccent OK")

from vosk import Model
m = Model(r"d:\Mycode\vosk-model-ru-0.42")
print("VOSK OK")