import argparse
import os
import pandas as pd
import requests
import json
from datetime import datetime
import re

# Функция для преобразования строкового представления размера в байты
def convert_size_to_bytes(size_str):
    if not isinstance(size_str, str):
        return None
    
    # Удаляем пробелы и приводим к нижнему регистру
    size_str = size_str.strip().lower()
    
    # Регулярное выражение для извлечения числа и единицы измерения
    pattern = r'([\d.]+)\s*([kmgtp]?b)'  # Поддерживает B, KB, MB, GB, TB, PB
    match = re.match(pattern, size_str)
    
    if not match:
        return None
    
    size_value = float(match.group(1))
    unit = match.group(2)
    
    # Преобразование в байты в зависимости от единицы измерения
    units = {'b': 1, 'kb': 1024, 'mb': 1024**2, 'gb': 1024**3, 'tb': 1024**4, 'pb': 1024**5}
    
    # Получаем множитель для единицы измерения
    multiplier = units.get(unit, 1)
    
    # Возвращаем размер в байтах
    return size_value * multiplier

# Функция для преобразования строкового представления размера в мегабайты
def convert_size_to_mb(size_str):
    if not isinstance(size_str, str):
        return None
    
    # Проверяем, если строка уже содержит числовое значение и единицу измерения в нашем формате
    if ' GB' in size_str or ' MB' in size_str:
        # Извлекаем числовое значение
        try:
            value = float(size_str.split(' ')[0])
            if ' GB' in size_str:
                # Преобразуем ГБ в МБ
                return value * 1024
            elif ' MB' in size_str:
                # Уже в МБ
                return value
        except (ValueError, IndexError):
            pass
    
    # Удаляем пробелы и приводим к нижнему регистру
    size_str = size_str.strip().lower()
    
    # Регулярное выражение для извлечения числа и единицы измерения
    pattern = r'([\d.]+)\s*([kmgtp]?b)'  # Поддерживает B, KB, MB, GB, TB, PB
    match = re.match(pattern, size_str)
    
    if not match:
        return None
    
    size_value = float(match.group(1))
    unit = match.group(2)
    
    # Преобразование в МБ в зависимости от единицы измерения
    if unit == 'b':
        return size_value / (1024 * 1024)  # Байты в МБ
    elif unit == 'kb':
        return size_value / 1024  # КБ в МБ
    elif unit == 'mb':
        return size_value  # Уже в МБ
    elif unit == 'gb':
        return size_value * 1024  # ГБ в МБ
    elif unit == 'tb':
        return size_value * 1024 * 1024  # ТБ в МБ
    elif unit == 'pb':
        return size_value * 1024 * 1024 * 1024  # ПБ в МБ
    else:
        return None  # Неизвестная единица измерения

# Функция для сбора данных о моделях
def collect_model_data(min_likes=150, max_models=1000, min_models=0, sort_by='likes', num_params=None, mem_size=None, div=None):
    # Проверяем, что указан только один из параметров фильтрации: num_params или mem_size
    if num_params is not None and mem_size is not None:
        print("ВНИМАНИЕ: Указаны оба параметра фильтрации (num_params и mem_size). Будет использован только num_params.")
        mem_size = None
    print(f"\nНачинаем сбор данных о моделях...")
    print(f"Параметры запроса: мин. лайков: {min_likes}, макс. моделей: {max_models}, мин. моделей: {min_models}, сортировка: {sort_by}")
    if num_params is not None:
        print(f"Фильтр по параметрам: {num_params * 100:.1f}% моделей" if num_params < 1 else f"Фильтр по параметрам: {num_params} млн {f'±{div} млн' if div is not None else ''}")
    if mem_size is not None:
        print(f"Фильтр по размеру: {mem_size} МБ {f'±{div} МБ' if div is not None else ''}")
    print("-" * 50)
    # Параметры для API запроса
    api_sort_by = 'likes'
    api_direction = -1  # По убыванию
    
    # Базовый URL для API
    base_url = 'https://huggingface.co/api/models'
    
    # Параметры запроса
    params = {
        'sort': api_sort_by,
        'direction': api_direction,
        'limit': max_models
    }
    
    # Выполняем запрос к API
    response = requests.get(base_url, params=params)
    
    # Проверяем успешность запроса
    if response.status_code != 200:
        print(f"Ошибка при запросе к API: {response.status_code}")
        return None
    
    # Получаем данные из ответа
    models = response.json()
    
    # Ограничиваем количество моделей
    models = models[:max_models]
    
    # Собираем данные о моделях
    data = []
    for model in models:
        # Проверяем наличие минимального количества лайков
        if 'likes' in model and model['likes'] < min_likes:
            continue
        
        # Получаем дополнительную информацию о модели
        model_info = get_model_info(model['modelId'])
        
        # Объединяем основную информацию с дополнительной
        model_data = {**model, **model_info}
        
        # Добавляем данные в список
        data.append(model_data)
    
    # Создаем DataFrame
    df = pd.DataFrame(data)
    
    print(f"Получено {len(df)} моделей из API")

    # Проверяем, достаточно ли моделей после всех фильтров
    if len(df) < min_models:
        print(f"ВНИМАНИЕ: После фильтрации осталось только {len(df)} моделей, что меньше требуемого минимума ({min_models}).")
        print("Попробуйте увеличить max_models или ослабить другие фильтры.")
        # Можно вернуть пустой DataFrame или текущий df, в зависимости от желаемого поведения
        # В данном случае, возвращаем текущий df, чтобы пользователь видел, что получилось


    
    # Преобразуем параметры в числовой формат для сортировки
    if 'parameters' in df.columns:
        df['parameters_numeric'] = pd.to_numeric(df['parameters'], errors='coerce')
    
    # Сортировка по лайкам и параметрам (если доступны)
    if 'parameters' in df.columns and 'likes' in df.columns:
        print(f"Сортировка по лайкам и параметрам (по убыванию)...")
        df = df.sort_values(by=['likes', 'parameters_numeric'], ascending=[False, False])
        print(f"Сортировка завершена. Топ 5 моделей:")
        for i, (idx, row) in enumerate(df.head(5).iterrows()):
            params_str = f", параметров: {row['parameters']}" if 'parameters' in row and pd.notna(row['parameters']) else ""
            print(f"  {i+1}. {row['modelId']}: лайков: {row['likes']}{params_str}")
    else:
        print(f"Сортировка по {sort_by} (по убыванию)...")
        df = df.sort_values(by=sort_by, ascending=False)
    
    # Фильтрация по размеру модели, если указан mem_size
    if mem_size is not None:
        print("\nПрименяем фильтрацию по размеру модели...")
        print(f"Размер всегда указывается в МБ (мегабайтах)")
        
        # Проверяем наличие поля model_size_mb
        if 'model_size_mb' in df.columns:
            # Используем напрямую числовое значение в МБ
            df['size_mb'] = pd.to_numeric(df['model_size_mb'], errors='coerce')
        elif 'model_size' in df.columns:
            # Преобразуем строковое представление размера в МБ
            df['size_mb'] = df['model_size'].apply(lambda x: convert_size_to_mb(x) if isinstance(x, str) else None)
        else:
            print("Предупреждение: Информация о размере модели недоступна. Фильтрация по размеру не будет применена.")
            return df
        
        original_count = len(df)
        
        # Если указан диапазон отклонения (div)
        if div is not None:
            min_size_mb = mem_size - div
            max_size_mb = mem_size + div
            
            print(f"Ищем модели с размером от {min_size_mb} до {max_size_mb} МБ...")
            df = df[(df['size_mb'] >= min_size_mb) & (df['size_mb'] <= max_size_mb)]
            print(f"Отфильтровано по размеру модели ({mem_size}±{div} МБ): осталось {len(df)} из {original_count} моделей")
        else:
            # Точный размер модели
            print(f"Ищем модели с точным размером {mem_size} МБ...")
            df = df[df['size_mb'] == mem_size]
            print(f"Отфильтровано по точному размеру модели ({mem_size} МБ): осталось {len(df)} из {original_count} моделей")
        
        # Удаляем временный столбец
        df = df.drop('size_mb', axis=1)
    
    
    # Фильтрация по количеству параметров
    if 'parameters' in df.columns and num_params is not None:
        print("\nПрименяем фильтрацию по параметрам модели...")
        original_count = len(df)
        
        # Преобразуем параметры в числовой формат
        def convert_params_to_numeric(param_str):
            if not param_str or not isinstance(param_str, str):
                return None
            
            # Удаляем пробелы и приводим к нижнему регистру
            param_str = param_str.strip().lower()
            
            # Проверяем на наличие единиц измерения
            if 'b' in param_str:
                # Значение в миллиардах, переводим в миллионы
                try:
                    return float(param_str.replace('b', '')) * 1000
                except ValueError:
                    return None
            elif 'm' in param_str:
                # Значение уже в миллионах
                try:
                    return float(param_str.replace('m', ''))
                except ValueError:
                    return None
            else:
                # Пробуем преобразовать как есть
                try:
                    return float(param_str)
                except ValueError:
                    return None
        
        df['parameters_numeric'] = df['parameters'].apply(convert_params_to_numeric)
        
        # Проверяем тип num_params
        is_percentage = isinstance(num_params, float) and num_params < 1
        
        # Если num_params < 1, то это процент моделей
        if is_percentage:
            print(f"Выбираем {num_params*100:.1f}% моделей с наибольшим количеством параметров...")
            # Сортируем по количеству параметров
            df = df.sort_values(by='parameters_numeric', ascending=False)
            
            # Вычисляем количество моделей для выбора
            num_models_to_select = max(1, int(len(df) * num_params))
            
            # Выбираем топ моделей
            df = df.head(num_models_to_select)
            
            print(f"Отфильтровано по проценту моделей с наибольшим количеством параметров ({num_params*100:.1f}%): осталось {len(df)} из {original_count} моделей")
        else:
            # Если указан диапазон отклонения (div)
            if div is not None:
                min_params = num_params - div
                max_params = num_params + div
                
                print(f"Ищем модели с параметрами от {min_params:.3f} до {max_params:.3f} млн...")
                df = df[(df['parameters_numeric'] >= min_params) & (df['parameters_numeric'] <= max_params)]
                print(f"Отфильтровано по количеству параметров ({num_params:.3f}±{div:.3f} млн): осталось {len(df)} из {original_count} моделей" )
            else:
                print(f"Ищем модели с точным количеством параметров {num_params:.3f} млн...")
                df = df[df['parameters_numeric'] == num_params]
                print(f"Отфильтровано по точному количеству параметров ({num_params:.3f} млн): осталось {len(df)} из {original_count} моделей")
        
        # Удаляем временный столбец
        df = df.drop('parameters_numeric', axis=1)
    
    # Формируем имя выходного файла
    output_filename = f"huggingface_models_{min_likes}likes"
    
    # Добавляем информацию о параметрах в имя файла
    if num_params is not None:
        if num_params < 1:
            output_filename += f"_top{int(num_params*100)}percent_params"
        else:
            if div is not None:
                output_filename += f"_{num_params:.1f}pm{div:.1f}M_params"
            else:
                output_filename += f"_{num_params:.1f}M_params"
    
    # Добавляем информацию о размере модели в имя файла
    if mem_size is not None:
        if div is not None:
            output_filename += f"_{mem_size}pm{div}MB_size"
        else:
            output_filename += f"_{mem_size}MB_size"
            
    # Добавляем расширение .csv к имени файла
    output_filename += ".csv"
    
    # Создаем директорию для сохранения результатов, если она не существует
    os.makedirs('output', exist_ok=True)
    
    # Сохраняем результаты в CSV файл
    output_path = os.path.join('output', output_filename)
    
    # Удаляем дубликаты перед сохранением
    if '_id' in df.columns:
        df = df.drop_duplicates(subset=['_id'])
    elif 'id' in df.columns:
        df = df.drop_duplicates(subset=['id'])
    elif 'modelId' in df.columns:
        df = df.drop_duplicates(subset=['modelId'])
    
    # Удаляем проблемные столбцы, которые могут вызывать проблемы с кодировкой
    problematic_columns = ['tags', 'model_type']
    for col in problematic_columns:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    # Сохраняем с правильной кодировкой
    df.to_csv(output_path, index=False, encoding='utf-8', lineterminator='\n')
    
    print(f"\nИтоговые результаты:")
    print(f"  Всего моделей после фильтрации: {len(df)}")
    if len(df) > 0:
        print(f"  Модель с наибольшим числом лайков: {df.iloc[0]['modelId']} ({df.iloc[0]['likes']} лайков)")
        if 'parameters_numeric' in df.columns and pd.notna(df.iloc[0]['parameters_numeric']):
            print(f"  Параметры этой модели: {df.iloc[0]['parameters_numeric']:.3f} млн")
    
    # Выводим список всех моделей после фильтрации
    print(f"\nСписок всех моделей после фильтрации:")
    for i, (idx, row) in enumerate(df.iterrows()):
        print(f"  {i+1}. {row['modelId']} (ID: {row['_id']})")
    
    # Проверяем, есть ли дубликаты по _id
    if '_id' in df.columns:
        duplicate_ids = df['_id'].duplicated().sum()
        print(f"\nКоличество дубликатов по _id: {duplicate_ids}")
    
    # Проверяем, есть ли дубликаты по modelId
    if 'modelId' in df.columns:
        duplicate_model_ids = df['modelId'].duplicated().sum()
        print(f"Количество дубликатов по modelId: {duplicate_model_ids}")
    print(f"\nДанные сохранены в файл: {output_path}")
    
    # Проверяем содержимое сохраненного файла
    try:
        df_saved = pd.read_csv(output_path)
        print(f"\nПроверка сохраненного файла:")
        print(f"  Количество строк в сохраненном файле: {len(df_saved)}")
        print(f"  Список моделей в сохраненном файле:")
        for i, row in df_saved.iterrows():
            print(f"    {i+1}. {row['modelId']}")
    except Exception as e:
        print(f"Ошибка при проверке сохраненного файла: {e}")
    
    return df

# Функция для получения дополнительной информации о модели
def get_model_info(model_id):
    # URL для получения информации о модели
    url = f'https://huggingface.co/api/models/{model_id}'
    
    # Выполняем запрос к API
    response = requests.get(url)
    
    # Проверяем успешность запроса
    if response.status_code != 200:
        print(f"Ошибка при запросе информации о модели {model_id}: {response.status_code}")
        return {}
    
    # Получаем данные из ответа
    model_info = response.json()
    
    # Извлекаем нужные поля
    info = {}
    
    # Извлекаем информацию из cardData, если она доступна
    card_data = model_info.get('cardData', {})
    
    # Извлекаем информацию из config, если она доступна
    config_data = model_info.get('config', {})
    
    # Извлекаем информацию из transformersInfo, если она доступна
    transformers_info = model_info.get('transformersInfo', {})
    
    # Собираем все возможные источники текста для извлечения информации
    sources = [
        model_info.get('modelId', ''),
        str(model_info.get('tags', [])),
    ]
    
    # Добавляем описание из cardData, если оно есть
    if isinstance(card_data, dict) and 'description' in card_data:
        sources.append(card_data.get('description', ''))
    
    # Добавляем информацию из config
    if isinstance(config_data, dict):
        sources.append(str(config_data))
    
    # Добавляем информацию из transformersInfo
    if isinstance(transformers_info, dict):
        sources.append(str(transformers_info))
    
    # Объединяем все источники в один текст
    combined_text = ' '.join(sources)
    
    # Проверяем наличие информации о параметрах в transformersInfo
    if isinstance(transformers_info, dict) and 'n_parameters' in transformers_info:
        try:
            param_value = float(transformers_info['n_parameters'])
            # Преобразуем в миллиарды для отображения
            info['parameters'] = param_value / 1000000000
            print(f"Извлечено из transformersInfo: параметры = {info['parameters']} млрд")
        except (ValueError, TypeError):
            pass
    
    # Проверяем наличие информации о параметрах в config
    if 'parameters' not in info and isinstance(config_data, dict):
        for key in ['num_parameters', 'n_parameters', 'params', 'parameters']:
            if key in config_data:
                try:
                    param_value = float(config_data[key])
                    # Преобразуем в миллиарды для отображения
                    info['parameters'] = param_value / 1000000000
                    print(f"Извлечено из config: параметры = {info['parameters']} млрд")
                    break
                except (ValueError, TypeError):
                    pass
    
    # Извлекаем количество параметров из названия модели
    if 'parameters' not in info:
        model_name = model_info.get('modelId', '')
        param_name_patterns = [
            r'(\d+\.?\d*)[bB]',  # Например, 7B, 13B, 1.5B
            r'(\d+\.?\d*)[mM]',  # Например, 125M, 350M
            r'-(\d+\.?\d*)[bB]',  # Например, llama-7b
            r'-(\d+\.?\d*)[mM]',  # Например, gpt-350m
        ]
        
        for pattern in param_name_patterns:
            matches = re.findall(pattern, model_name, re.IGNORECASE)
            if matches:
                try:
                    param_value = float(matches[0])
                    if 'b' in pattern.lower():
                        # Миллиарды параметров
                        info['parameters'] = param_value  # Оставляем как есть (в миллиардах)
                    else:
                        # Миллионы параметров
                        info['parameters'] = param_value / 1000  # Преобразуем в миллиарды
                    print(f"Извлечено из имени модели: параметры = {info['parameters']} млрд")
                    break
                except (ValueError, IndexError):
                    continue
    
    # Извлекаем количество параметров из текста
    if 'parameters' not in info:
        # Регулярные выражения для поиска информации о параметрах
        param_patterns = [
            r'(\d+\.?\d*)\s*[bB]illion\s*parameters',
            r'(\d+\.?\d*)\s*[bB]\s*parameters',
            r'(\d+\.?\d*)\s*[bB]\s*param',
            r'(\d+\.?\d*)\s*[bB]',
            r'-(\d+\.?\d*)[bB]',
            r'(\d+\.?\d*)[bB]',
            r'(\d+)\s*parameters',
            r'(\d+)\s*param',
            r'(\d+\.?\d*)\s*million\s*parameters',
            r'(\d+\.?\d*)\s*[mM]\s*parameters',
            r'(\d+\.?\d*)\s*[mM]\s*param',
            r'(\d+\.?\d*)\s*[mM]',
            r'parameters:\s*(\d+\.?\d*)',
            r'params:\s*(\d+\.?\d*)',
        ]
        
        # Пытаемся извлечь количество параметров
        for pattern in param_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            if matches:
                try:
                    # Возвращаем первое найденное значение
                    param_value = float(matches[0])
                    if 'billion' in pattern.lower() or 'b' in pattern.lower():
                        # Миллиарды параметров
                        info['parameters'] = param_value  # Оставляем как есть (в миллиардах)
                    elif 'million' in pattern.lower() or 'm' in pattern.lower():
                        # Миллионы параметров
                        info['parameters'] = param_value / 1000  # Преобразуем в миллиарды
                    else:
                        # Если единица не указана явно, делаем предположение на основе значения
                        if param_value < 100:
                            # Вероятно, это миллиарды параметров
                            info['parameters'] = param_value
                        elif param_value < 1000000:
                            # Вероятно, это миллионы параметров
                            info['parameters'] = param_value / 1000
                        else:
                            # Вероятно, это абсолютное количество параметров
                            info['parameters'] = param_value / 1000000000
                    
                    print(f"Извлечено из текста: параметры = {info['parameters']} млрд")
                    break
                except (ValueError, KeyError):
                    continue
    
    # Проверяем наличие информации о размере в config
    if isinstance(config_data, dict) and ('model_size' in config_data or 'size' in config_data):
        size_key = 'model_size' if 'model_size' in config_data else 'size'
        try:
            size_value = float(config_data[size_key])
            # Предполагаем, что размер указан в МБ
            info['model_size'] = f"{size_value} MB"
            info['model_size_mb'] = size_value
            print(f"Извлечено из config: размер модели = {info['model_size']}")
        except (ValueError, TypeError):
            pass
    
    # Извлекаем размер модели из текста
    if 'model_size' not in info:
        # Регулярные выражения для поиска информации о размере модели
        size_patterns = [
            r'(\d+\.?\d*)\s*[gG][bB]',
            r'(\d+\.?\d*)\s*[gG]',
            r'(\d+\.?\d*)\s*[mM][bB]',
            r'(\d+\.?\d*)\s*[mM]',
            r'size:\s*(\d+\.?\d*)\s*[gG][bB]',
            r'size:\s*(\d+\.?\d*)\s*[mM][bB]',
            r'model size:\s*(\d+\.?\d*)\s*[gG][bB]',
            r'model size:\s*(\d+\.?\d*)\s*[mM][bB]',
        ]
        
        # Пытаемся извлечь размер модели
        for pattern in size_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            if matches:
                try:
                    # Возвращаем первое найденное значение
                    size_value = float(matches[0])
                    
                    # Определяем единицу измерения на основе шаблона
                    if 'gb' in pattern.lower() or 'g' in pattern.lower():
                        # Гигабайты
                        info['model_size'] = f"{size_value} GB"
                        info['model_size_mb'] = size_value * 1024  # Преобразуем в МБ для фильтрации
                    elif 'mb' in pattern.lower() or 'm' in pattern.lower():
                        # Мегабайты
                        info['model_size'] = f"{size_value} MB"
                        info['model_size_mb'] = size_value
                    
                    print(f"Извлечено: размер модели = {info['model_size']}")
                    break
                except (ValueError, KeyError):
                    continue
    
    # Проверяем наличие информации о размере в siblings
    if 'model_size' not in info and 'siblings' in model_info:
        siblings = model_info.get('siblings', [])
        total_size = 0
        
        for sibling in siblings:
            if isinstance(sibling, dict) and 'rfilename' in sibling and 'size' in sibling:
                # Учитываем только файлы моделей (safetensors, bin, pt, ckpt)
                filename = sibling.get('rfilename', '').lower()
                if any(ext in filename for ext in ['.safetensors', '.bin', '.pt', '.ckpt', '.model']):
                    total_size += sibling.get('size', 0)
        
        if total_size > 0:
            # Преобразуем байты в МБ
            size_mb = total_size / (1024 * 1024)
            if size_mb > 1024:
                # Если больше 1 ГБ, отображаем в ГБ
                size_gb = size_mb / 1024
                info['model_size'] = f"{size_gb:.2f} GB"
            else:
                info['model_size'] = f"{size_mb:.2f} MB"
            
            info['model_size_mb'] = size_mb
            print(f"Извлечено из siblings: размер модели = {info['model_size']}")
    
    # Добавляем больше полезной информации о модели
    # Тип модели (архитектура)
    if 'pipeline_tag' in model_info:
        info['model_type'] = model_info['pipeline_tag']
    
    # Теги модели
    if 'tags' in model_info:
        info['tags'] = ', '.join(model_info['tags'])
    
    # Дата последнего обновления
    if 'lastModified' in model_info:
        info['last_modified'] = model_info['lastModified']
    
    # Количество загрузок
    if 'downloads' in model_info:
        info['downloads'] = model_info['downloads']
    
    # Автор модели
    if 'author' in model_info:
        info['author'] = model_info['author']
    
    # Лицензия
    if 'license' in model_info:
        info['license'] = model_info['license']
    
    # Язык модели (если указан)
    if 'language' in model_info:
        info['language'] = model_info['language']
    
    # Библиотека, с которой работает модель
    if 'library_name' in model_info:
        info['library'] = model_info['library_name']
    
    # Краткое описание (если есть)
    if isinstance(card_data, dict) and 'description' in card_data:
        description = card_data['description']
        if len(description) > 200:
            description = description[:197] + '...'
        info['description'] = description
    
    # Добавляем числовое значение параметров для фильтрации
    if 'parameters' in info:
        info['parameters_numeric'] = float(info['parameters'])
        # Форматируем строковое представление параметров
        if info['parameters'] >= 1:
            info['parameters'] = f"{info['parameters']:.1f}B"
        else:
            info['parameters'] = f"{info['parameters']*1000:.1f}M"
    
    return info

# Функция для запуска сбора данных с параметрами командной строки
def main():
    # Создаем парсер аргументов командной строки
    parser = argparse.ArgumentParser(description='Сбор данных о моделях с Hugging Face')
    parser.add_argument('--min_likes', type=int, default=150, help='Минимальное количество лайков')
    parser.add_argument('--max_models', type=int, default=1000, help='Максимальное количество моделей для сбора')
    parser.add_argument('--min_models', type=int, default=0, help='Минимальное количество моделей для сохранения')
    parser.add_argument('--sort_by', type=str, default='likes',
                        choices=['likes', 'parameters'], 
                        help='Параметр для сортировки (примечание: всегда выполняется сортировка по лайкам и параметрам)')
    parser.add_argument('--num_params', type=float, default=None, 
                        help='Количество параметров модели (в миллионах) или процент моделей (если < 1)')
    parser.add_argument('--mem_size', type=float, default=None, 
                        help='Размер модели в МБ')
    parser.add_argument('--div', type=float, default=None, 
                        help='Диапазон отклонения для num_params и mem_size')
    
    args = parser.parse_args()
    
    # Проверяем взаимоисключающие параметры
    if args.num_params is not None and args.mem_size is not None:
        print("ВНИМАНИЕ: Указаны оба параметра фильтрации (num_params и mem_size). Будет использован только num_params.")
        args.mem_size = None
    
    # Выводим информацию о параметрах запуска
    print(f"Запуск сбора данных с параметрами:")
    print(f"  Минимальное количество лайков: {args.min_likes}")
    print(f"  Максимальное количество моделей: {args.max_models}")
    print(f"  Минимальное количество моделей: {args.min_models}")
    print(f"  Сортировка по: {args.sort_by}")
    
    if args.num_params is not None:
        if args.num_params < 1:
            print(f"  Фильтрация по проценту моделей с наибольшим количеством параметров: {args.num_params*100}%")
        else:
            if args.div is not None:
                print(f"  Фильтрация по количеству параметров: {args.num_params}±{args.div} млн")
            else:
                print(f"  Фильтрация по точному количеству параметров: {args.num_params} млн")
    
    if args.mem_size is not None:
        if args.div is not None:
            print(f"  Фильтрация по размеру модели: {args.mem_size}±{args.div} МБ")
        else:
            print(f"  Фильтрация по размеру модели: {args.mem_size} МБ")
    
    # Запускаем сбор данных
    collect_model_data(
        min_likes=args.min_likes,
        max_models=args.max_models,
        min_models=args.min_models,
        sort_by=args.sort_by,
        num_params=args.num_params,
        mem_size=args.mem_size,
        div=args.div
    )

if __name__ == '__main__':
    main()