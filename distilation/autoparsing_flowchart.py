import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle
import matplotlib.colors as mcolors

# Настройка стиля
plt.style.use('seaborn-v0_8-whitegrid')

# Создаем фигуру для блок-схемы автопарсинга моделей
fig, ax = plt.subplots(figsize=(14, 10))

# Определяем цвета
colors = {
    'start_end': '#4CAF50',  # зеленый
    'process': '#2196F3',     # синий
    'decision': '#FF9800',    # оранжевый
    'data': '#9C27B0',        # фиолетовый
    'arrow': '#555555',       # серый
    'background': '#F5F5F5',  # светло-серый
    'text': '#212121'         # темно-серый
}

# Устанавливаем фон
ax.set_facecolor(colors['background'])
fig.patch.set_facecolor(colors['background'])

# Функция для создания блока
def create_block(x, y, width, height, color, text, fontsize=10):
    rect = Rectangle((x, y), width, height, facecolor=color, edgecolor=colors['arrow'], 
                     alpha=0.8, linewidth=2, zorder=1)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', 
            color=colors['text'], fontsize=fontsize, fontweight='bold', wrap=True)

# Функция для создания ромба (решение)
def create_decision(x, y, width, height, text, fontsize=10):
    points = np.array([
        [x, y + height/2],
        [x + width/2, y],
        [x + width, y + height/2],
        [x + width/2, y + height]
    ])
    polygon = plt.Polygon(points, facecolor=colors['decision'], edgecolor=colors['arrow'], 
                         alpha=0.8, linewidth=2, zorder=1)
    ax.add_patch(polygon)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', 
            color=colors['text'], fontsize=fontsize, fontweight='bold', wrap=True)

# Функция для создания стрелки
def create_arrow(start, end, text=None, fontsize=8):
    arrow = FancyArrowPatch(start, end, color=colors['arrow'], linewidth=1.5,
                          arrowstyle='-|>', connectionstyle='arc3,rad=0.1', zorder=0)
    ax.add_patch(arrow)
    if text:
        # Вычисляем середину стрелки
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        # Добавляем небольшое смещение для текста
        offset_x = (end[0] - start[0]) * 0.1
        offset_y = (end[1] - start[1]) * 0.1
        ax.text(mid_x + offset_x, mid_y + offset_y, text, ha='center', va='center', 
                color=colors['text'], fontsize=fontsize, fontweight='bold', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

# Создаем блоки для автопарсинга моделей
# Начало
create_block(2, 9, 3, 0.8, colors['start_end'], 'Начало', fontsize=12)

# Ввод параметров
create_block(2, 8, 3, 0.8, colors['data'], 'Ввод параметров учителя\nи коэффициента сжатия', fontsize=11)

# Расчет целевых параметров
create_block(2, 7, 3, 0.8, colors['process'], 'Расчет целевого количества\nпараметров студента', fontsize=11)

# Определение разброса
create_decision(2, 5.8, 3, 0.8, 'Разброс указан?', fontsize=11)

# Расчет разброса по умолчанию
create_block(6, 5.8, 3, 0.8, colors['process'], 'Расчет разброса\nпо умолчанию (20%)', fontsize=11)

# Поиск моделей
create_block(2, 4.6, 3, 0.8, colors['process'], 'Поиск моделей на\nHugging Face', fontsize=11)

# Проверка количества моделей
create_decision(2, 3.4, 3, 0.8, 'Найдено достаточно\nмоделей?', fontsize=11)

# Расширение диапазона поиска
create_block(6, 3.4, 3, 0.8, colors['process'], 'Увеличение разброса\nи снижение требований', fontsize=11)

# Повторная проверка
create_decision(6, 2.2, 3, 0.8, 'Найдено достаточно\nмоделей?', fontsize=11)

# Вывод топ моделей
create_block(2, 1, 3, 0.8, colors['process'], 'Вывод топ-5 моделей\nс параметрами', fontsize=11)

# Проверка автовыбора
create_decision(2, -0.2, 3, 0.8, 'Автоматический\nвыбор?', fontsize=11)

# Автовыбор первой модели
create_block(6, -0.2, 3, 0.8, colors['process'], 'Выбор первой модели\nиз списка', fontsize=11)

# Запрос выбора пользователя
create_block(2, -1.4, 3, 0.8, colors['data'], 'Запрос выбора\nмодели у пользователя', fontsize=11)

# Возврат выбранной модели
create_block(2, -2.6, 3, 0.8, colors['process'], 'Возврат выбранной\nмодели студента', fontsize=11)

# Конец
create_block(2, -3.8, 3, 0.8, colors['start_end'], 'Конец', fontsize=12)

# Ошибка - модели не найдены
create_block(10, 2.2, 3, 0.8, colors['process'], 'Вывод сообщения\nоб ошибке', fontsize=11)

# Создаем стрелки
create_arrow((3.5, 9.4), (3.5, 9))
create_arrow((3.5, 8), (3.5, 7.8))
create_arrow((3.5, 7), (3.5, 6.6))
create_arrow((3.5, 5.8), (3.5, 5.4), 'Да')
create_arrow((5, 6.2), (6, 6.2), 'Нет')
create_arrow((6, 5.8), (6, 5.4))
create_arrow((6, 5.4), (3.5, 5.4))
create_arrow((3.5, 5.4), (3.5, 5))
create_arrow((3.5, 4.6), (3.5, 4.2))
create_arrow((3.5, 3.4), (3.5, 3), 'Да')
create_arrow((5, 3.8), (6, 3.8), 'Нет')
create_arrow((6, 3.4), (6, 3))
create_arrow((6, 2.2), (6, 1.8), 'Да')
create_arrow((6, 1.8), (3.5, 1.8))
create_arrow((3.5, 1.8), (3.5, 1.8))
create_arrow((3.5, 1), (3.5, 0.6))
create_arrow((3.5, -0.2), (3.5, -0.6), 'Нет')
create_arrow((5, 0.2), (6, 0.2), 'Да')
create_arrow((6, -0.2), (6, -0.6))
create_arrow((6, -0.6), (3.5, -0.6))
create_arrow((3.5, -0.6), (3.5, -1.4))
create_arrow((3.5, -1.4), (3.5, -1.8))
create_arrow((3.5, -2.6), (3.5, -3))
create_arrow((3.5, -3.8), (3.5, -4.2))
create_arrow((9, 2.6), (10, 2.6), 'Нет')
create_arrow((10, 2.2), (10, 1.8))
create_arrow((10, 1.8), (3.5, -3))

# Настройка осей
ax.set_xlim(0, 14)
ax.set_ylim(-4.5, 10)
ax.axis('off')

# Заголовок
ax.set_title('Блок-схема автопарсинга моделей ученика для дистилляции', fontsize=16, fontweight='bold', pad=20)

# Сохраняем блок-схему
plt.tight_layout()
plt.savefig('autoparsing_flowchart.png', dpi=300, bbox_inches='tight')
plt.close()

print("Блок-схема автопарсинга моделей ученика создана и сохранена как 'autoparsing_flowchart.png'")