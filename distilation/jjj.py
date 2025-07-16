from transformers import AutoTokenizer, AutoModelForCausalLM

teacher_tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
student_tokenizer = AutoTokenizer.from_pretrained("gpt2")
print("Teacher vocab size:", teacher_tokenizer.vocab_size)
print("Student vocab size:", student_tokenizer.vocab_size)