import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Путь к сохраненной модели
MODEL_PATH = "./output/distillation#1_15.07.25.11.38.56/student_model_wikitext.pt"  # Укажите точное имя файла
MODEL_NAME = "arnir0/Tiny-LLM"
TOKENIZER_NAME = "arnir0/Tiny-LLM"  # Используйте "arnir0/Tiny-LLM", если словарь не расширялся

# Загрузка токенизатора
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# Загрузка базовой архитектуры модели
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Загрузка state_dict
state_dict = torch.load(MODEL_PATH, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"), weights_only=True)
model.load_state_dict(state_dict)

# Перевод модели в режим оценки и на устройство
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Если словарь расширен до 50257, обновите конфигурацию
if TOKENIZER_NAME == "gpt2-xl":
    model.config.vocab_size = 50257

def generate_text(prompt, model, tokenizer, max_length=512, temperature=1.0, top_k=50, top_p=0.95):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def main():
    prompt = "According to all known laws of aviation, there is no way a bee should be able to fly."
    generated_text = generate_text(prompt, model, tokenizer)
    print("Generated text:", generated_text)

if __name__ == "__main__":
    main()