from distillation_logic import distill_gpt2_wikitext

def distill_models(
    teacher_model='gpt2-xl',
    student_model='arnir0/Tiny-LLM',
    box_type='white',
    teacher_model_path=None,
    student_model_path=None,
    dataset='wikitext',
    dataset_path=None,
    sample_size=1000,
    max_length=256,
    temperature=2.0,
    alpha=0.7,
    beta=0.3,
    epochs=2,
    batch_size=8,
    learning_rate=5e-5,
    no_cuda=False,
    output_dir='./output',
    folder_name=None,
    checkpoint_interval=50,
    save_metrics=True,
    save_plot=True,
    save_best_model=True,
    skip_validation=False,
    install_deps=True,
    run_tests=False,
    skip_demo=True
):
    """
    Функция дистилляции знаний между двумя моделями
    
    Args:
        teacher_model (str): Название модели учителя
        student_model (str): Название модели студента
        box_type (str): Тип дистилляции ('white' или 'black')
        teacher_model_path (str): Путь к локальной модели учителя
        student_model_path (str): Путь к локальной модели студента
        dataset (str): Название датасета
        dataset_path (str): Путь к пользовательскому датасету
        sample_size (int): Размер выборки из датасета
        max_length (int): Максимальная длина последовательности
        temperature (float): Температура для дистилляции
        alpha (float): Вес для soft targets
        beta (float): Вес для hard targets
        epochs (int): Количество эпох обучения
        batch_size (int): Размер батча
        learning_rate (float): Скорость обучения
        no_cuda (bool): Не использовать CUDA
        output_dir (str): Директория для сохранения результатов
        folder_name (str): Название папки для результатов
        checkpoint_interval (int): Интервал сохранения чекпоинтов
        save_metrics (bool): Сохранить метрики в JSON
        save_plot (bool): Сохранить график потерь
        save_best_model (bool): Сохранять лучшую модель
        skip_validation (bool): Пропустить валидацию
        install_deps (bool): Установить зависимости
        run_tests (bool): Запустить тесты
        skip_demo (bool): Пропустить демонстрацию
    
    Returns:
        tuple: (обученная модель студента, метрики дистилляции)
    """
    
    class Args:
        def __init__(self):
            self.teacher_model = teacher_model
            self.student_model = student_model
            self.box = box_type
            self.teacher_model_path = teacher_model_path
            self.student_model_path = student_model_path
            self.dataset = dataset
            self.dataset_path = dataset_path
            self.sample_size = sample_size
            self.max_length = max_length
            self.temperature = temperature
            self.alpha = alpha
            self.beta = beta
            self.epochs = epochs
            self.batch_size = batch_size
            self.lr = learning_rate
            self.no_cuda = no_cuda
            self.output_dir = output_dir
            self.folder_name = folder_name
            self.checkpoint_interval = checkpoint_interval
            self.save_metrics = save_metrics
            self.save_plot = save_plot
            self.save_best_model = save_best_model
            self.skip_validation = skip_validation
            self.install_deps = install_deps
            self.run_tests = run_tests
            self.skip_demo = skip_demo

    args = Args()
    return distill_gpt2_wikitext(args)

# Вызов функции
trained_student, metrics = distill_models()