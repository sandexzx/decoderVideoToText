import os
import glob
import subprocess
import time
import sys
from pathlib import Path
import whisper
from tqdm import tqdm
import threading

# Необходимые библиотеки:
# pip install openai-whisper ffmpeg-python tqdm

def get_video_duration(video_path):
    """
    Получает длительность видео в секундах с помощью ffprobe
    """
    command = [
        "ffprobe", 
        "-v", "error", 
        "-show_entries", "format=duration", 
        "-of", "default=noprint_wrappers=1:nokey=1", 
        video_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return float(result.stdout.strip())
    except (ValueError, IndexError):
        return 0

def extract_audio(video_path, output_audio_path):
    """
    Извлекает аудио дорожку из видео файла с помощью ffmpeg с отображением прогресса
    """
    print(f"🎬 Подготовка к извлечению аудио из {video_path}...")
    
    # Получаем длительность видео
    duration = get_video_duration(video_path)
    if duration <= 0:
        print("⚠️ Не удалось определить продолжительность видео. Прогресс не будет отображаться.")
        # Если не удалось определить длительность, просто запускаем ffmpeg без отслеживания прогресса
        command = [
            "ffmpeg", 
            "-i", video_path,
            "-q:a", "0",
            "-map", "a",
            "-y",
            output_audio_path
        ]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        # Создаем прогресс-бар
        progress_bar = tqdm(total=100, desc="Извлечение аудио", 
                          unit="%", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}")
        
        # Команда для получения временных меток ffmpeg
        command = [
            "ffmpeg", 
            "-i", video_path,
            "-q:a", "0",
            "-map", "a",
            "-y",
            "-progress", "pipe:1",  # Вывод прогресса в stdout
            output_audio_path
        ]
        
        # Запускаем процесс
        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            universal_newlines=True
        )
        
        # Отслеживаем прогресс
        for line in process.stdout:
            if line.startswith("out_time_ms="):
                try:
                    time_ms = float(line.split("=")[1].strip())
                    current_time = time_ms / 1000000  # Конвертируем микросекунды в секунды
                    progress = min(int((current_time / duration) * 100), 100)
                    progress_bar.update(progress - progress_bar.n)
                except (ValueError, IndexError):
                    pass
        
        # Завершаем процесс и закрываем прогресс-бар
        process.wait()
        progress_bar.close()
    
    # Проверяем, существует ли файл
    if os.path.exists(output_audio_path):
        print(f"✅ Аудио успешно извлечено в {output_audio_path}")
    else:
        print(f"❌ Ошибка при извлечении аудио.")
        sys.exit(1)

def estimate_audio_duration(audio_path):
    """
    Оценивает длительность аудио файла для расчета прогресса
    """
    command = [
        "ffprobe", 
        "-v", "error", 
        "-show_entries", "format=duration", 
        "-of", "default=noprint_wrappers=1:nokey=1", 
        audio_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return float(result.stdout.strip())
    except (ValueError, IndexError):
        return 0

class TranscriptionProgressTracker:
    """
    Класс для отслеживания прогресса транскрибирования без использования callback
    """
    def __init__(self, audio_duration):
        self.audio_duration = audio_duration
        self.progress_bar = None
        self.running = True
        self.temp_file = None
        self.last_size = 0
        
    def start_tracking(self, model, temp_file):
        """
        Запускает отслеживание прогресса в отдельном потоке
        """
        self.temp_file = temp_file
        self.progress_bar = tqdm(total=100, desc="Распознавание речи", 
                               unit="%", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}")
        
        # Оцениваем скорость обработки на основе первых нескольких секунд
        try:
            # Предполагаем, что файл модели whisper_state.csv будет создан во временной директории
            thread = threading.Thread(target=self._track_progress)
            thread.daemon = True
            thread.start()
        except Exception as e:
            print(f"⚠️ Ошибка при отслеживании прогресса: {e}")
            return None
        
        return self.progress_bar
    
    def _track_progress(self):
        """
        Отслеживает прогресс, проверяя созданные временные файлы или состояние Whisper
        """
        # Начальное значение прогресса
        progress = 0
        last_update = time.time()
        
        while self.running and progress < 100:
            time.sleep(0.5)  # Проверяем прогресс каждые полсекунды
            
            # Если есть временный файл, проверяем его размер
            if self.temp_file and os.path.exists(self.temp_file):
                current_size = os.path.getsize(self.temp_file)
                if current_size > self.last_size:
                    # Увеличиваем прогресс на основе изменения размера файла
                    self.last_size = current_size
                    progress += 1
                    self.progress_bar.update(1)
            else:
                # Если нет временного файла, используем временную эвристику
                current_time = time.time()
                if current_time - last_update >= 1.0:  # Обновляем каждую секунду
                    # Используем простую эвристику для прогресса
                    progress += 1
                    self.progress_bar.update(1 if progress <= 95 else 0)  # Останавливаемся на 95%
                    last_update = current_time
            
            # Предотвращаем превышение 100%
            if progress > 100:
                progress = 100
                
        # Завершаем прогресс-бар при выходе из цикла
        if self.progress_bar:
            self.progress_bar.update(100 - self.progress_bar.n)  # Достигаем 100%
    
    def stop(self):
        """
        Останавливает отслеживание прогресса
        """
        self.running = False
        if self.progress_bar:
            self.progress_bar.close()

def transcribe_audio(audio_path, output_text_path):
    """
    Транскрибирует аудио файл с помощью локальной модели Whisper (medium)
    с эмуляцией отслеживания прогресса
    """
    print(f"🎤 Загружаем модель Whisper (medium)...")
    
    # Отображаем прогресс загрузки модели
    with tqdm(total=1, desc="Загрузка модели", unit="шаг") as progress_bar:
        # Загружаем medium модель
        model = whisper.load_model("medium")
        progress_bar.update(1)
    
    print(f"🎧 Начинаем распознавание текста из аудио {audio_path}...")
    
    # Оценка длительности аудио для отслеживания прогресса
    audio_duration = estimate_audio_duration(audio_path)
    
    # Создаем временный файл для отслеживания прогресса
    temp_file = f"{output_text_path}.tmp"
    if os.path.exists(temp_file):
        os.remove(temp_file)
        
    # Запускаем трекер прогресса
    progress_tracker = TranscriptionProgressTracker(audio_duration)
    progress_tracker.start_tracking(model, temp_file)
    
    try:
        # Выполняем транскрибирование
        result = model.transcribe(audio_path, verbose=False)
        
        # Создаем временный файл для отслеживания
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write("1" * 100)  # Просто для имитации прогресса
            
        # Получаем текст из результата
        transcription = result["text"]
        
        # Сохраняем результат в текстовый файл
        with open(output_text_path, "w", encoding="utf-8") as text_file:
            text_file.write(transcription)
            
        # Останавливаем отслеживание прогресса
        progress_tracker.stop()
        
        # Удаляем временный файл
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        print(f"✅ Текст успешно распознан и сохранен в {output_text_path}")
        return transcription
    
    except Exception as e:
        progress_tracker.stop()
        print(f"❌ Ошибка при распознавании: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise

def main():
    start_time = time.time()
    
    print("🔍 Ищем MP4 файл в текущей директории...")
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
    
    # Извлечение аудио с отображением прогресса
    extract_audio(video_path, audio_path)
    
    # Транскрибирование аудио с эмуляцией отображения прогресса
    transcription = transcribe_audio(audio_path, text_path)
    
    # Вычисляем общее время выполнения
    total_time = time.time() - start_time
    minutes, seconds = divmod(total_time, 60)
    
    print("\n" + "="*60)
    print(f"🎉 Задача выполнена за {int(minutes)} мин {int(seconds)} сек!")
    print(f"📂 Исходный файл: {video_path}")
    print(f"🎵 Извлеченное аудио: {audio_path}")
    print(f"📝 Файл транскрипции: {text_path}")
    print(f"💾 Размер файла транскрипции: {os.path.getsize(text_path)} байт")
    print("-"*60)
    print("📌 Первые 150 символов транскрипции:")
    print(transcription[:150] + "..." if len(transcription) > 150 else transcription)
    print("="*60)

if __name__ == "__main__":
    main()
