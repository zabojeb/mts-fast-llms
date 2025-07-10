import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
import time
from dataclasses import dataclass


@dataclass
class DistillationConfig:
    """Конфигурация для дистилляции знаний"""
    temperature: float = 4.0
    alpha: float = 0.7  # Вес для soft targets
    beta: float = 0.3   # Вес для hard targets
    log_interval: int = 100
    save_metrics: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class DistillationMetrics:
    """Класс для сбора и хранения метрик дистилляции"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_loss = 0.0
        self.distillation_loss = 0.0
        self.student_loss = 0.0
        self.teacher_loss = 0.0  # Добавлено
        self.num_batches = 0
        self.start_time = time.time()

    def update(self, total_loss: float, distillation_loss: float, student_loss: float, teacher_loss: float):
        self.total_loss += total_loss
        self.distillation_loss += distillation_loss
        self.student_loss += student_loss
        self.teacher_loss += teacher_loss  # Добавлено
        self.num_batches += 1

    def get_average_metrics(self) -> Dict[str, float]:
        if self.num_batches == 0:
            return {}

        return {
            'avg_total_loss': self.total_loss / self.num_batches,
            'avg_distillation_loss': self.distillation_loss / self.num_batches,
            'avg_student_loss': self.student_loss / self.num_batches,
            'avg_teacher_loss': self.teacher_loss / self.num_batches,  # Добавлено
            'elapsed_time': time.time() - self.start_time
        }


class BaseOptimizer(ABC):
    """Базовый класс для оптимизаторов (квантизация, дистилляция и т.д.)"""
    
    @abstractmethod
    def apply(self, model: nn.Module, *args, **kwargs) -> nn.Module:
        """Применить оптимизацию к модели"""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Получить метрики оптимизации"""
        pass


class KnowledgeDistillation(BaseOptimizer):
    """Модуль для дистилляции знаний"""
    
    def __init__(self, config: DistillationConfig):
        self.config = config
        self.metrics = DistillationMetrics()
        self.logger = self._setup_logger()
        self.loss_history = {'total': [], 'distill': [], 'student': [], 'teacher': []}
        
        self.teacher_model: Optional[nn.Module] = None
        self.student_model: Optional[nn.Module] = None
        
    def _setup_logger(self) -> logging.Logger:
        """Настройка логгера"""
        logger = logging.getLogger('distillation')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def set_models(self, teacher_model: nn.Module, student_model: nn.Module):
        """Установить модели учителя и студента"""
        self.teacher_model = teacher_model.to(self.config.device)
        self.student_model = student_model.to(self.config.device)
        
        # Переводим учителя в режим eval
        self.teacher_model.eval()
        
        self.logger.info(f"Модели установлены на устройство: {self.config.device}")
        self.logger.info(f"Параметры учителя: {sum(p.numel() for p in self.teacher_model.parameters()):,}")
        self.logger.info(f"Параметры студента: {sum(p.numel() for p in self.student_model.parameters()):,}")
    
    def distillation_loss(
        self, 
        student_logits: torch.Tensor, 
        teacher_logits: torch.Tensor, 
        targets: torch.Tensor,
        student_loss_fn: Callable = F.cross_entropy
    ) -> Dict[str, torch.Tensor]:
        """
        Вычисление функции потерь для дистилляции.

        Основная часть формулы — это KL-дивергенция между
        распределениями студента и учителя при температуре `T`:
        `T^2 * KL(student_logits / T || teacher_logits / T)` с
        параметром `reduction='batchmean'`. Затем добавляется обычная
        cross-entropy по истинным меткам.

        Args:
            student_logits: Логиты студента
            teacher_logits: Логиты учителя
            targets: Истинные метки
            student_loss_fn: Функция потерь для студента

        Returns:
            Словарь с компонентами потерь
        """
        # Soft targets (дистилляция)
        teacher_probs = F.softmax(teacher_logits / self.config.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / self.config.temperature, dim=-1)
        
        # KL дивергенция вычисляется с учётом температурного масштаба.
        # Формула: T^2 * KL(student_log_probs || teacher_probs)
        distillation_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (self.config.temperature ** 2)
        
        # Hard targets (обычная потеря)
        student_loss = student_loss_fn(student_logits, targets)
        
        # Общая потеря
        total_loss = (
            self.config.alpha * distillation_loss + 
            self.config.beta * student_loss
        )

        # Потери учителя (для сравнения)
        with torch.no_grad():
            teacher_loss = student_loss_fn(teacher_logits, targets)
        
        return {
            'total_loss': total_loss,
            'distillation_loss': distillation_loss,
            'student_loss': student_loss,
            'teacher_loss': teacher_loss
        }
    
    def train_step(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        student_loss_fn: Callable = F.cross_entropy
    ) -> Dict[str, float]:
        """
        Один шаг обучения с дистилляцией
        
        Args:
            inputs: Входные данные
            targets: Целевые метки
            optimizer: Оптимизатор для студента
            student_loss_fn: Функция потерь для студента
        
        Returns:
            Метрики шага обучения
        """
        if self.teacher_model is None or self.student_model is None:
            raise ValueError("Модели не установлены. Используйте set_models()")
        
        inputs = inputs.to(self.config.device)
        targets = targets.to(self.config.device)
        
        # Получаем предсказания учителя (без градиентов)
        with torch.no_grad():
            teacher_logits = self.teacher_model(inputs)
        
        # Получаем предсказания студента
        self.student_model.train()
        student_logits = self.student_model(inputs)
        
        # Вычисляем потери
        losses = self.distillation_loss(
            student_logits, teacher_logits, targets, student_loss_fn
        )
        
        # Обратное распространение
        optimizer.zero_grad()
        losses['total_loss'].backward()
        optimizer.step()
        
        # Обновляем метрики
        self.metrics.update(
            losses['total_loss'].item(),
            losses['distillation_loss'].item(),
            losses['student_loss'].item(),
            losses['teacher_loss'].item()
        )

        # Сохраняем историю для графика
        self.loss_history['total'].append(losses['total_loss'].item())
        self.loss_history['distill'].append(losses['distillation_loss'].item())
        self.loss_history['student'].append(losses['student_loss'].item())
        self.loss_history['teacher'].append(losses['teacher_loss'].item())
        
        return {
            'total_loss': losses['total_loss'].item(),
            'distillation_loss': losses['distillation_loss'].item(),
            'student_loss': losses['student_loss'].item(),
            'teacher_loss': losses['teacher_loss'].item()
        }
    
    def apply(
        self, 
        student_model: nn.Module, 
        teacher_model: nn.Module,
        train_loader: Any,
        optimizer: torch.optim.Optimizer,
        num_epochs: int = 1,
        student_loss_fn: Callable = F.cross_entropy,
        checkpoint_dir: str = None,
        checkpoint_interval: int = 1
    ) -> nn.Module:
        """
        Применить дистилляцию к модели студента
        
        Args:
            student_model: Модель студента
            teacher_model: Модель учителя
            train_loader: Загрузчик данных для обучения
            optimizer: Оптимизатор
            num_epochs: Количество эпох
            student_loss_fn: Функция потерь для студента
            checkpoint_dir: Директория для сохранения чекпоинтов
            checkpoint_interval: Интервал сохранения чекпоинтов (в эпохах)
        
        Returns:
            Обученная модель студента
        """
        self.set_models(teacher_model, student_model)
        self.metrics.reset()
        
        self.logger.info(f"Начало дистилляции на {num_epochs} эпох")
        self.logger.info(f"Конфигурация: T={self.config.temperature}, α={self.config.alpha}, β={self.config.beta}")
        
        # Создаем директорию для чекпоинтов, если она указана
        if checkpoint_dir:
            import os
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        total_batches = len(train_loader)
        
        for epoch in range(num_epochs):
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                step_metrics = self.train_step(inputs, targets, optimizer, student_loss_fn)
                
                # Логирование
                if (batch_idx + 1) % self.config.log_interval == 0:
                    self.logger.info(
                        f"Эпоха {epoch+1}/{num_epochs}, Батч {batch_idx + 1}/{total_batches}: "
                        f"Total: {step_metrics['total_loss']:.4f}, "
                        f"Distill: {step_metrics['distillation_loss']:.4f}, "
                        f"Student: {step_metrics['student_loss']:.4f}, "
                        f"Teacher: {step_metrics['teacher_loss']:.4f}"
                    )
            
            # Сохраняем чекпоинт, если указан интервал и директория
            if checkpoint_dir and (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, 'student_checkpoint.pt')
                self.save_student_model(checkpoint_path)
                self.logger.info(f"Сохранен чекпоинт после эпохи {epoch+1} в {checkpoint_path}")
            
            avg_metrics = self.metrics.get_average_metrics()
            self.logger.info(
                f"Эпоха {epoch+1}/{num_epochs} завершена. Средние потери: "
                f"Student: {avg_metrics['avg_student_loss']:.4f}, "
                f"Total: {avg_metrics['avg_total_loss']:.4f}, "
                f"Distill: {avg_metrics['avg_distillation_loss']:.4f}"
            )
        
        self.logger.info("Дистилляция завершена")
        return self.student_model
    
    def get_metrics(self) -> Dict[str, Any]:
        """Получить метрики дистилляции"""
        base_metrics = self.metrics.get_average_metrics()
        
        config_info = {
            'temperature': self.config.temperature,
            'alpha': self.config.alpha,
            'beta': self.config.beta,
            'device': self.config.device
        }
        
        if self.teacher_model and self.student_model:
            model_info = {
                'teacher_params': sum(p.numel() for p in self.teacher_model.parameters()),
                'student_params': sum(p.numel() for p in self.student_model.parameters()),
                'compression_ratio': sum(p.numel() for p in self.teacher_model.parameters()) / 
                                   sum(p.numel() for p in self.student_model.parameters())
            }
            config_info.update(model_info)
        
        return {
            'metrics': base_metrics,
            'config': config_info
        }

    def get_loss_history(self) -> Dict[str, list]:
        """Возвращает историю потерь по батчам."""
        return self.loss_history
    
    def save_student_model(self, path: str):
        """Сохранить обученную модель студента"""
        if self.student_model is None:
            raise ValueError("Модель студента не установлена")
        
        torch.save({
            'model_state_dict': self.student_model.state_dict(),
            'config': self.config,
            'metrics': self.get_metrics()
        }, path)
        
        self.logger.info(f"Модель студента сохранена в {path}")


# Пример использования совместимого интерфейса
class QuantizationOptimizer(BaseOptimizer):
    """Пример класса квантизации с совместимым интерфейсом"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics = {}
    
    def apply(self, model: nn.Module, *args, **kwargs) -> nn.Module:
        # Здесь была бы логика квантизации
        return model
    
    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics


def create_optimizer(optimizer_type: str, config: Dict[str, Any]) -> BaseOptimizer:
    """Фабричная функция для создания оптимизаторов"""
    if optimizer_type == "distillation":
        return KnowledgeDistillation(DistillationConfig(**config))
    elif optimizer_type == "quantization":
        return QuantizationOptimizer(config)
    else:
        raise ValueError(f"Неизвестный тип оптимизатора: {optimizer_type}")