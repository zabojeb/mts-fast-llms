import os
import subprocess
import sys

def run_script(script_path):
    """Запускает Python скрипт и выводит его результат"""
    print(f"Запуск {os.path.basename(script_path)}...")
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при запуске {os.path.basename(script_path)}:")
        print(e.stderr)
        return False

def main():
    """Запускает все скрипты для создания блок-схем"""
    print("Генерация блок-схем для дистилляции...\n")
    
    # Пути к скриптам
    current_dir = os.path.dirname(os.path.abspath(__file__))
    scripts = [
        os.path.join(current_dir, "autoparsing_flowchart.py"),
        os.path.join(current_dir, "whitebox_distillation_flowchart.py"),
        os.path.join(current_dir, "blackbox_distillation_flowchart.py")
    ]
    
    # Запуск каждого скрипта
    success_count = 0
    for script in scripts:
        if run_script(script):
            success_count += 1
        print("\n" + "-"*50 + "\n")
    
    # Вывод итогов
    print(f"Генерация завершена: {success_count}/{len(scripts)} блок-схем создано успешно.")
    
    # Проверка наличия файлов
    output_files = [
        "autoparsing_flowchart.png",
        "whitebox_distillation_flowchart.png",
        "blackbox_distillation_flowchart.png"
    ]
    
    print("\nСозданные файлы:")
    for file in output_files:
        file_path = os.path.join(current_dir, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / 1024  # размер в КБ
            print(f"  ✓ {file} ({size:.1f} КБ)")
        else:
            print(f"  ✗ {file} (не создан)")

if __name__ == "__main__":
    main()