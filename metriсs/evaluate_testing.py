from datasets import load_dataset

# Загружаем датасет
dataset = load_dataset("fka/awesome-chatgpt-prompts", split="train[:2]")  # Берем только 2 примера для скорости

# Проверяем структуру датасета
print(dataset)