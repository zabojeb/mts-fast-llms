import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle
import matplotlib.colors as mcolors

# Настройка стиля
plt.style.use('seaborn-v0_8-whitegrid')

# Создаем фигуру для блок-схемы black-box дистилляции
fig, ax = plt.subplots(figsize=(14, 12))

# Определяем цвета
colors = {
    'start_end': '#4CAF50',  # зеленый
    'process': '#2196F3',     # синий
    'decision': '#FF9800',    # оранжевый
    'data': '#9C27B0',        # фиолетовый
    'arrow': '#555555',       # серый
    'background': '#F5F5F5',  # светло-серый
    'text': '#212121',        # темно-серый
    'teacher': '#E91E63',     # розовый
    'student': '#00BCD4'      # голубой
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

# Создаем блоки для Black-Box дистилляции
# Начало
create_block(5, 10, 3, 0.8, colors['start_end'], 'Начало Black-Box Дистилляции', fontsize=12)

# Инициализация моделей
create_block(2, 9, 3, 0.8, colors['teacher'], 'Инициализация\nмодели учителя', fontsize=11)
create_block(8, 9, 3, 0.8, colors['student'], 'Инициализация\nмодели ученика', fontsize=11)

# Загрузка данных
create_block(5, 8, 3, 0.8, colors['data'], 'Загрузка обучающих\nи валидационных данных', fontsize=11)

# Настройка оптимизатора
create_block(5, 7, 3, 0.8, colors['process'], 'Настройка оптимизатора\nи планировщика', fontsize=11)

# Цикл по эпохам
create_block(5, 6, 3, 0.8, colors['process'], 'Начало цикла\nпо эпохам', fontsize=11)

# Цикл по батчам
create_block(5, 5, 3, 0.8, colors['process'], 'Начало цикла\nпо батчам', fontsize=11)

# Получение выходов учителя
create_block(2, 4, 3, 0.8, colors['teacher'], 'Получение предсказаний\nот модели учителя', fontsize=11)

# Получение выходов ученика
create_block(8, 4, 3, 0.8, colors['student'], 'Получение логитов\nот модели ученика', fontsize=11)

# Расчет потерь
create_block(2, 3, 3, 0.8, colors['process'], 'Расчет потери\nдистилляции (KL)', fontsize=11)
create_block(8, 3, 3, 0.8, colors['process'], 'Расчет потери\nученика (CE)', fontsize=11)

# Комбинирование потерь
create_block(5, 2, 3, 0.8, colors['process'], 'Комбинирование потерь\nс весами alpha и beta', fontsize=11)

# Обратное распространение
create_block(5, 1, 3, 0.8, colors['process'], 'Обратное распространение\nи обновление весов', fontsize=11)

# Конец цикла по батчам
create_decision(5, 0, 3, 0.8, 'Все батчи\nобработаны?', fontsize=11)

# Валидация
create_block(5, -1, 3, 0.8, colors['process'], 'Валидация модели\nна тестовых данных', fontsize=11)

# Проверка ранней остановки
create_decision(5, -2, 3, 0.8, 'Критерий ранней\nостановки?', fontsize=11)

# Конец цикла по эпохам
create_decision(5, -3, 3, 0.8, 'Все эпохи\nзавершены?', fontsize=11)

# Сохранение модели
create_block(5, -4, 3, 0.8, colors['student'], 'Сохранение\nмодели ученика', fontsize=11)

# Конец
create_block(5, -5, 3, 0.8, colors['start_end'], 'Конец', fontsize=12)

# Создаем стрелки
create_arrow((6.5, 10.4), (6.5, 10))
create_arrow((3.5, 9.4), (3.5, 9))
create_arrow((9.5, 9.4), (9.5, 9))
create_arrow((3.5, 9), (3.5, 8.4))
create_arrow((9.5, 9), (9.5, 8.4))
create_arrow((3.5, 8.4), (5, 8.4))
create_arrow((9.5, 8.4), (8, 8.4))
create_arrow((6.5, 8), (6.5, 7.8))
create_arrow((6.5, 7), (6.5, 6.8))
create_arrow((6.5, 6), (6.5, 5.8))
create_arrow((6.5, 5), (6.5, 4.8))
create_arrow((6.5, 4.8), (3.5, 4.8))
create_arrow((6.5, 4.8), (9.5, 4.8))
create_arrow((3.5, 4), (3.5, 3.8))
create_arrow((9.5, 4), (9.5, 3.8))
create_arrow((3.5, 3), (3.5, 2.4))
create_arrow((9.5, 3), (9.5, 2.4))
create_arrow((3.5, 2.4), (5, 2.4))
create_arrow((9.5, 2.4), (8, 2.4))
create_arrow((6.5, 2), (6.5, 1.8))
create_arrow((6.5, 1), (6.5, 0.8))
create_arrow((6.5, 0), (6.5, -0.2), 'Да')
create_arrow((8, 0.4), (9.5, 0.4), 'Нет')
create_arrow((9.5, 0.4), (9.5, 5.4))
create_arrow((9.5, 5.4), (6.5, 5.4))
create_arrow((6.5, -1), (6.5, -1.2))
create_arrow((6.5, -2), (6.5, -2.2), 'Да')
create_arrow((8, -1.6), (9.5, -1.6), 'Нет')
create_arrow((6.5, -3), (6.5, -3.2), 'Да')
create_arrow((8, -2.6), (9.5, -2.6), 'Нет')
create_arrow((9.5, -2.6), (9.5, 6.4))
create_arrow((9.5, 6.4), (6.5, 6.4))
create_arrow((6.5, -4), (6.5, -4.2))
create_arrow((6.5, -5), (6.5, -5.2))

# Добавляем пояснения
ax.text(2, 2.4, 'KL дивергенция между\nлог-вероятностями ученика\nи предсказаниями учителя', ha='center', va='center', 
        color=colors['text'], fontsize=8, fontweight='bold', 
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

ax.text(8, 2.4, 'Cross-Entropy между\nпредсказаниями ученика\nи истинными метками', ha='center', va='center', 
        color=colors['text'], fontsize=8, fontweight='bold', 
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

ax.text(5, 2.8, 'Loss = α * Distillation_Loss + β * Student_Loss', ha='center', va='center', 
        color=colors['text'], fontsize=9, fontweight='bold', 
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

# Добавляем ключевое отличие от White-Box
ax.text(2, 4.4, 'Только выходные\nпредсказания, без\nдоступа к логитам', ha='center', va='center', 
        color='red', fontsize=9, fontweight='bold', 
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='red', pad=1))

# Настройка осей
ax.set_xlim(0, 12)
ax.set_ylim(-5.5, 11)
ax.axis('off')

# Заголовок
ax.set_title('Блок-схема Black-Box дистилляции', fontsize=16, fontweight='bold', pad=20)

# Сохраняем блок-схему
plt.tight_layout()
plt.savefig('blackbox_distillation_flowchart.png', dpi=300, bbox_inches='tight')
plt.close()

print("Блок-схема Black-Box дистилляции создана и сохранена как 'blackbox_distillation_flowchart.png'")