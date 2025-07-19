from distillation_logic import distill_gpt2_wikitext
from types import SimpleNamespace

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
    args = SimpleNamespace(
        teacher_model=teacher_model,
        student_model=student_model,
        box=box_type,  # Исправлено: box_type -> box
        teacher_model_path=teacher_model_path,
        student_model_path=student_model_path,
        dataset=dataset,
        dataset_path=dataset_path,
        sample_size=sample_size,
        max_length=max_length,
        temperature=temperature,
        alpha=alpha,
        beta=beta,
        epochs=epochs,
        batch_size=batch_size,
        lr=learning_rate,
        no_cuda=no_cuda,
        output_dir=output_dir,
        folder_name=folder_name,
        checkpoint_interval=checkpoint_interval,
        save_metrics=save_metrics,
        save_plot=save_plot,
        save_best_model=save_best_model,
        skip_validation=skip_validation,
        install_deps=install_deps,
        run_tests=run_tests,
        skip_demo=skip_demo
    )
    return distill_gpt2_wikitext(args)

distill_models()