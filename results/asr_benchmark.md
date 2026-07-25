# Сравнение распознавания речи

WER = (замены + удаления + вставки) / число слов эталона. Ниже — лучше.
Эталон используется только для измерения после распознавания; Whisper
получает пустой prompt и работает как в режиме свободной речи.

| Запись | Движок | Слов | S | D | I | WER |
|---|---|---:|---:|---:|---:|---:|
| 1local | vosk | 110 | 46 | 0 | 14 | 54.55% |
| 1local | whisper-large-v3-turbo | 110 | 29 | 2 | 5 | 32.73% |
| 2local | vosk | 209 | 17 | 5 | 6 | 13.40% |
| 2local | whisper-large-v3-turbo | 209 | 11 | 2 | 11 | 11.48% |
| 3local | vosk | 101 | 4 | 0 | 2 | 5.94% |
| 3local | whisper-large-v3-turbo | 101 | 0 | 0 | 1 | 0.99% |
| 4local | vosk | 162 | 11 | 2 | 1 | 8.64% |
| 4local | whisper-large-v3-turbo | 162 | 6 | 0 | 4 | 6.17% |
| 5local | vosk | 261 | 41 | 9 | 4 | 20.69% |
| 5local | whisper-large-v3-turbo | 261 | 13 | 3 | 5 | 8.05% |
| 6local | vosk | 185 | 16 | 2 | 4 | 11.89% |
| 6local | whisper-large-v3-turbo | 185 | 4 | 0 | 4 | 4.32% |
| 1fori | vosk | 123 | 40 | 2 | 12 | 43.90% |
| 1fori | whisper-large-v3-turbo | 123 | 17 | 2 | 0 | 15.45% |
| 2fori | vosk | 111 | 51 | 0 | 17 | 61.26% |
| 2fori | whisper-large-v3-turbo | 111 | 18 | 0 | 4 | 19.82% |
| 4fori | vosk | 75 | 39 | 1 | 60 | 133.33% |
| 4fori | whisper-large-v3-turbo | 75 | 22 | 1 | 23 | 61.33% |
| 5fori | vosk | 76 | 18 | 0 | 5 | 30.26% |
| 5fori | whisper-large-v3-turbo | 76 | 5 | 0 | 0 | 6.58% |
| 6fori | vosk | 43 | 8 | 0 | 1 | 20.93% |
| 6fori | whisper-large-v3-turbo | 43 | 5 | 1 | 0 | 13.95% |
| 7fori | vosk | 42 | 4 | 0 | 3 | 16.67% |
| 7fori | whisper-large-v3-turbo | 42 | 1 | 0 | 0 | 2.38% |
| 8fori | vosk | 43 | 5 | 0 | 3 | 18.60% |
| 8fori | whisper-large-v3-turbo | 43 | 1 | 0 | 0 | 2.33% |

## Micro-average

| Движок | Записей | Слов | Ошибок | WER |
|---|---:|---:|---:|---:|
| vosk | 13 | 1541 | 453 | 29.40% |
| whisper-large-v3-turbo | 13 | 1541 | 200 | 12.98% |
