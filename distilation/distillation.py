import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Optional, Callable, Union, List
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


class WhiteBoxDistillation(BaseOptimizer):
    """Модуль для white-box дистилляции знаний (с доступом к логитам учителя)"""
    
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
        Вычисление функции потерь для дистилляции
        
        Args:
            student_logits: Логиты студента
            teacher_logits: Логиты учителя
            targets: Истинные метки
            student_loss_fn: Функция потерь для студента
        
        Returns:
            Словарь с компонентами потерь
        """
        # Soft targets (дистилляция)
        teacher_probs = F.softmax(teacher_logits / self.config.temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / self.config.temperature, dim=1)
        
        distillation_loss = F.kl_div(
            student_log_probs, 
            teacher_probs, 
            reduction='mean'
        )
        
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
class BlackBoxDistillation(BaseOptimizer):
    """Модуль для black-box дистилляции знаний (без доступа к логитам учителя)
    
    В отличие от white-box дистилляции, black-box дистилляция использует только
    выходы модели учителя (предсказания), а не внутренние логиты. Это полезно,
    когда у нас нет доступа к внутренней структуре модели учителя или когда
    архитектуры учителя и студента сильно различаются.
    
    Для защиты от переобучения используется ранняя остановка на основе валидационной выборки.
    """
    
    def __init__(self, config: DistillationConfig):
        self.config = config
        self.metrics = DistillationMetrics()
        self.logger = self._setup_logger()
        self.loss_history = {'total': [], 'distill': [], 'student': [], 'teacher': []}
        
        self.teacher_model: Optional[nn.Module] = None
        self.student_model: Optional[nn.Module] = None
        
        # Для ранней остановки
        self.best_val_loss = float('inf')
        self.patience = 3  # Количество эпох без улучшения
        self.patience_counter = 0
        self.best_model_state = None
        
    def _setup_logger(self) -> logging.Logger:
        """Настройка логгера"""
        logger = logging.getLogger('black_box_distillation')
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
        teacher_outputs: torch.Tensor, 
        targets: torch.Tensor,
        student_loss_fn: Callable = F.cross_entropy
    ) -> Dict[str, torch.Tensor]:
        """
        Вычисление функции потерь для black-box дистилляции
        
        Args:
            student_logits: Логиты студента
            teacher_outputs: Выходы учителя (предсказания, не логиты)
            targets: Истинные метки
            student_loss_fn: Функция потерь для студента
        
        Returns:
            Словарь с компонентами потерь
        """
        # Soft targets (дистилляция)
        # В black-box дистилляции мы используем предсказания учителя как мягкие метки
        student_log_probs = F.log_softmax(student_logits / self.config.temperature, dim=1)
        
        # KL-дивергенция между предсказаниями студента и учителя
        distillation_loss = F.kl_div(
            student_log_probs, 
            teacher_outputs, 
            reduction='mean'
        )
        
        # Hard targets (обычная потеря)
        student_loss = student_loss_fn(student_logits, targets)
        
        # Общая потеря
        total_loss = (
            self.config.alpha * distillation_loss + 
            self.config.beta * student_loss
        )

        # Потери учителя (для сравнения)
        with torch.no_grad():
            teacher_logits = torch.log(teacher_outputs + 1e-10) * self.config.temperature
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
        Один шаг обучения с black-box дистилляцией
        
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
            teacher_probs = F.softmax(teacher_logits / self.config.temperature, dim=1)
        
        # Получаем предсказания студента
        self.student_model.train()
        student_logits = self.student_model(inputs)
        
        # Вычисляем потери
        losses = self.distillation_loss(
            student_logits, teacher_probs, targets, student_loss_fn
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
    
    def validate(
        self,
        val_loader: Any,
        student_loss_fn: Callable = F.cross_entropy
    ) -> float:
        """
        Валидация модели студента
        
        Args:
            val_loader: Загрузчик данных для валидации
            student_loss_fn: Функция потерь для студента
            
        Returns:
            Средняя потеря на валидационной выборке
        """
        if self.student_model is None:
            raise ValueError("Модель студента не установлена. Используйте set_models()")
        
        self.student_model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.config.device)
                targets = targets.to(self.config.device)
                
                student_logits = self.student_model(inputs)
                loss = student_loss_fn(student_logits, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return avg_loss
    
    def apply(
        self, 
        student_model: nn.Module, 
        teacher_model: nn.Module,
        train_loader: Any,
        optimizer: torch.optim.Optimizer,
        num_epochs: int = 1,
        student_loss_fn: Callable = F.cross_entropy,
        checkpoint_dir: str = None,
        checkpoint_interval: int = 1,
        val_loader: Any = None,  # Добавлен валидационный загрузчик для ранней остановки
        patience: int = 3  # Количество эпох без улучшения для ранней остановки
    ) -> nn.Module:
        """
        Применить black-box дистилляцию к модели студента
        
        Args:
            student_model: Модель студента
            teacher_model: Модель учителя
            train_loader: Загрузчик данных для обучения
            optimizer: Оптимизатор
            num_epochs: Количество эпох
            student_loss_fn: Функция потерь для студента
            checkpoint_dir: Директория для сохранения чекпоинтов
            checkpoint_interval: Интервал сохранения чекпоинтов (в эпохах)
            val_loader: Загрузчик данных для валидации (для ранней остановки)
            patience: Количество эпох без улучшения для ранней остановки
        
        Returns:
            Обученная модель студента
        """
        self.set_models(teacher_model, student_model)
        self.metrics.reset()
        self.patience = patience
        self.patience_counter = 0
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        self.logger.info(f"Начало black-box дистилляции на {num_epochs} эпох")
        self.logger.info(f"Конфигурация: T={self.config.temperature}, α={self.config.alpha}, β={self.config.beta}")
        
        # Создаем директорию для чекпоинтов, если она указана
        if checkpoint_dir:
            import os
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        total_batches = len(train_loader)
        
        for epoch in range(num_epochs):
            # Обучение
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
            
            # Валидация и ранняя остановка
            if val_loader is not None:
                val_loss = self.validate(val_loader, student_loss_fn)
                self.logger.info(f"Эпоха {epoch+1}/{num_epochs}, Валидационная потеря: {val_loss:.4f}")
                
                # Проверяем, улучшилась ли модель
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.best_model_state = self.student_model.state_dict().copy()
                    self.logger.info(f"Новая лучшая валидационная потеря: {val_loss:.4f}")
                else:
                    self.patience_counter += 1
                    self.logger.info(f"Валидационная потеря не улучшилась. Терпение: {self.patience_counter}/{self.patience}")
                    
                    if self.patience_counter >= self.patience:
                        self.logger.info(f"Ранняя остановка на эпохе {epoch+1}")
                        # Восстанавливаем лучшую модель
                        if self.best_model_state is not None:
                            self.student_model.load_state_dict(self.best_model_state)
                        break
            
            # Сохраняем чекпоинт, если указан интервал и директория
            if checkpoint_dir and (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'student_checkpoint_epoch_{epoch+1}.pt')
                self.save_student_model(checkpoint_path)
                self.logger.info(f"Сохранен чекпоинт после эпохи {epoch+1} в {checkpoint_path}")
            
            avg_metrics = self.metrics.get_average_metrics()
            self.logger.info(
                f"Эпоха {epoch+1}/{num_epochs} завершена. Средние потери: "
                f"Student: {avg_metrics['avg_student_loss']:.4f}, "
                f"Total: {avg_metrics['avg_total_loss']:.4f}, "
                f"Distill: {avg_metrics['avg_distillation_loss']:.4f}"
            )
        
        # Если у нас есть лучшая модель из валидации, используем ее
        if val_loader is not None and self.best_model_state is not None:
            self.student_model.load_state_dict(self.best_model_state)
            self.logger.info("Загружена лучшая модель по валидационной потере")
        
        self.logger.info("Black-box дистилляция завершена")
        return self.student_model
    
    def get_metrics(self) -> Dict[str, Any]:
        """Получить метрики дистилляции"""
        base_metrics = self.metrics.get_average_metrics()
        
        config_info = {
            'temperature': self.config.temperature,
            'alpha': self.config.alpha,
            'beta': self.config.beta,
            'device': self.config.device,
            'distillation_type': 'black-box'
        }
        
        if self.teacher_model and self.student_model:
            model_info = {
                'teacher_params': sum(p.numel() for p in self.teacher_model.parameters()),
                'student_params': sum(p.numel() for p in self.student_model.parameters()),
                'compression_ratio': sum(p.numel() for p in self.teacher_model.parameters()) / 
                                sum(p.numel() for p in self.student_model.parameters())
            }
            config_info.update(model_info)
        
        # Добавляем информацию о ранней остановке
        if hasattr(self, 'best_val_loss'):
            config_info['best_val_loss'] = self.best_val_loss
        
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


def create_optimizer(optimizer_type: str, config: Dict[str, Any], box_type: str = "white") -> BaseOptimizer:
    """Фабричная функция для создания оптимизаторов
    
    Args:
        optimizer_type: Тип оптимизатора ('distillation', 'quantization')
        config: Конфигурация оптимизатора
        box_type: Тип дистилляции ('white' или 'black')
    
    Returns:
        Экземпляр оптимизатора
    """
    if optimizer_type == "distillation":
        if box_type.lower() == "white":
            return WhiteBoxDistillation(DistillationConfig(**config))
        elif box_type.lower() == "black":
            return BlackBoxDistillation(DistillationConfig(**config))
        else:
            raise ValueError(f"Неизвестный тип дистилляции: {box_type}. Используйте 'white' или 'black'")
    elif optimizer_type == "quantization":
        return QuantizationOptimizer(config)
    else:
        raise ValueError(f"Неизвестный тип оптимизатора: {optimizer_type}")