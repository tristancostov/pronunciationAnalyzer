import json
import wave
import os
import struct


with open("./analysis/4local_syllable_analysis.json", "r", encoding="utf-8") as f:
    report = json.load(f)


wf       = wave.open("./audio/4local.wav", "rb")
sr       = wf.getframerate()
nch      = wf.getnchannels()
sampW    = wf.getsampwidth()
allBytes = wf.readframes(wf.getnframes())
wf.close()


os.makedirs("syllables_export8", exist_ok=True)

count = 0
allWords  = report["wordAnalysis"]
startIdx  = len(allWords) // 2  
for wordItem in allWords[startIdx:]:
    word     = wordItem["word"]
    wordStart = wordItem["start"]   

    for sylItem in wordItem["syllableAnalysis"]:
        if count >= 25:
            break

        syl      = sylItem["syllable"]
        
        absStart = wordStart + sylItem["startSec"]
        absEnd   = wordStart + sylItem["endSec"]

        startSample = int(absStart * sr)
        endSample   = int(absEnd   * sr)

        
        bytesPerSample = sampW * nch
        startByte      = startSample * bytesPerSample
        endByte        = endSample   * bytesPerSample
        chunk          = allBytes[startByte:endByte]

        if len(chunk) < 100:
            continue

        fname = f"syllables_export8/{count+1:02d}_{word}_{syl}.wav"
        out   = wave.open(fname, "wb")
        out.setnchannels(nch)
        out.setsampwidth(sampW)
        out.setframerate(sr)
        out.writeframes(chunk)
        out.close()

        print(f"[{count+1:02d}] {word} → [{syl}]  {absStart:.3f}s – {absEnd:.3f}s  → {fname}")
        count += 1

    if count >= 25:
        break

print(f"\n完成！共切出 {count} 个音节，保存在 syllables_export8/ 文件夹里")