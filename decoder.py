import os
import glob
import subprocess
from pathlib import Path
import whisper

# Для начала нам нужно установить необходимые библиотеки, если их нет:
# pip install openai-whisper ffmpeg-python

def extract_audio(video_path, output_audio_path):
    """
    Извлекает аудио дорожку из видео файла с помощью ffmpeg
    """
    print(f"🎬 Извлекаем аудио из {video_path}...")
    # Используем subprocess для запуска ffmpeg
    command = [
        "ffmpeg", 
        "-i", video_path,  # входной файл
        "-q:a", "0",       # качество аудио (0 - лучшее)
        "-map", "a",       # выбираем только аудио поток
        "-y",              # перезаписывать существующий файл
        output_audio_path  # выходной аудио файл
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"✅ Аудио успешно извлечено в {output_audio_path}")

def transcribe_audio(audio_path, output_text_path):
    """
    Транскрибирует аудио файл с помощью локальной модели Whisper (medium)
    """
    print(f"🎤 Загружаем модель Whisper (medium)...")
    # Загружаем medium модель (это займет некоторое время при первом запуске)
    model = whisper.load_model("medium")
    
    print(f"🎧 Начинаем распознавание текста из аудио {audio_path}...")
    # Выполняем транскрибирование
    result = model.transcribe(audio_path)
    
    # Получаем текст из результата
    transcription = result["text"]
    
    # Сохраняем результат в текстовый файл
    with open(output_text_path, "w", encoding="utf-8") as text_file:
        text_file.write(transcription)
    
    print(f"✅ Текст успешно распознан и сохранен в {output_text_path}")
    return transcription

def main():
    # Найти единственный mp4 файл в текущей директории
    mp4_files = glob.glob("*.mp4")
    
    if not mp4_files:
        print("❌ Ошибка: MP4 файл не найден в текущей директории!")
        return
    
    if len(mp4_files) > 1:
        print(f"⚠️ Внимание: Найдено несколько MP4 файлов. Будет использован первый: {mp4_files[0]}")
    
    video_path = mp4_files[0]
    base_name = Path(video_path).stem
    
    # Пути для выходных файлов
    audio_path = f"{base_name}.mp3"
    text_path = f"{base_name}_transcription.txt"
    
    # Извлечение аудио
    extract_audio(video_path, audio_path)
    
    # Транскрибирование аудио
    transcription = transcribe_audio(audio_path, text_path)
    
    print("\n" + "="*50)
    print(f"🎉 Готово! Транскрипция сохранена в файл {text_path}")
    print(f"💾 Размер файла транскрипции: {os.path.getsize(text_path)} байт")
    print(f"📝 Первые 150 символов транскрипции:")
    print("-"*50)
    print(transcription[:150] + "..." if len(transcription) > 150 else transcription)
    print("="*50)

if __name__ == "__main__":
    main()
