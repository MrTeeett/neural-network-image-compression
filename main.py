"""
Модуль для обучения моделей сжатия и реконструкции изображений с использованием нейросетей ComCNN и RecCNN.

В данном модуле реализованы следующие компоненты:
1. Функция get_image_paths для сбора путей к изображениям в каталоге.
2. Функция augment_image для выполнения аугментаций над изображениями.
3. Класс ImageDataset для создания датасета изображений с возможностью трансформаций и изменения размера.
4. Класс ComCNN – сверточная сеть для сжатия изображений.
5. Класс ResidualBlock – простой residual-блок для улучшения архитектуры.
6. Класс RecCNN – сеть для реконструкции изображений, использующая residual-блоки.
7. Функция approx_codec для имитации блочных артефактов квантования за счёт добавления шума.
8. Функция combined_loss для вычисления комбинированной функции потерь (MSE + MS-SSIM).
9. Функция train_model для обучения моделей с подробным циклом обучения и валидации, включая сравнение с JPEG-сжатием.
10. Функция main – основная точка входа, собирающая данные, создающая датасеты и запускающая обучение.

Автор: Ваша команда
"""

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Библиотека для MS-SSIM (pip install pytorch-msssim)
from pytorch_msssim import ms_ssim
from io import BytesIO  # Импорт для работы с in-memory буфером


def get_image_paths(root_dir):
    """
    Проходит по заданному каталогу и собирает пути к изображениям с расширениями .png, .jpg, .jpeg.
    
    Args:
        root_dir (str): Путь к корневому каталогу, в котором будет осуществляться поиск изображений.
    
    Returns:
        list: Список строк, каждая из которых является полным путем к найденному изображению.
    
    Функция рекурсивно проходит по подкаталогам с использованием os.walk.
    """
    image_extensions = ('.png', '.jpg', '.jpeg')
    image_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(image_extensions):
                image_paths.append(os.path.join(dirpath, f))
    return image_paths


def augment_image(img):
    """
    Применяет случайные аугментации к изображению.
    
    Args:
        img (numpy.ndarray): Массив изображения в оттенках серого с размерностью [H, W] и значениями в диапазоне [0, 1].
    
    Returns:
        numpy.ndarray: Аугментированное изображение.
    
    Действия, выполняемые функцией:
      - С вероятностью 0.5 выполняется горизонтальное отражение.
      - С вероятностью 0.5 выполняется вертикальное отражение.
      - Производится поворот изображения на случайное кратное 90 градусов.
    """
    # Горизонтальное отражение
    if random.random() > 0.5:
        img = np.fliplr(img)
    # Вертикальное отражение
    if random.random() > 0.5:
        img = np.flipud(img)
    # Случайный поворот (0, 90, 180 или 270 градусов)
    rotations = random.choice([0, 1, 2, 3])
    if rotations:
        img = np.rot90(img, rotations)
    return img.copy()


class ImageDataset(Dataset):
    """
    Кастомный датасет для загрузки изображений в оттенках серого с применением трансформаций и изменением размера.
    
    Атрибуты:
        image_files (list): Список путей к изображениям.
        transform (callable, optional): Функция или трансформация для применения к изображениям после загрузки.
        target_size (tuple): Желаемый размер выходного изображения (ширина, высота).
    
    Методы:
        __len__(): Возвращает общее количество изображений.
        __getitem__(idx): Загружает изображение по индексу, изменяет размер и применяет трансформацию.
    
    Поддерживается инициализация либо путем к каталогу (str), либо списком путей.
    При передаче каталога, датасет автоматически ищет файлы с расширениями .png, .jpg и .jpeg.
    """
    def __init__(self, data, transform=None, target_size=(256, 256)):
        """
        Инициализирует датасет.
        
        Args:
            data (str or list): Путь к каталогу с изображениями или список путей.
            transform (callable, optional): Функция для трансформации изображения.
            target_size (tuple, optional): Размер, к которому будет приведено изображение (по умолчанию (256, 256)).
        """
        super(ImageDataset, self).__init__()
        if isinstance(data, str):
            try:
                self.image_files = [os.path.join(data, f) for f in os.listdir(data)
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            except Exception as e:
                print(f"Ошибка при получении списка файлов в каталоге {data}: {e}")
                self.image_files = []
        elif isinstance(data, list):
            self.image_files = data
        else:
            raise ValueError("data должен быть либо строкой (путь к папке), либо списком путей")
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        """
        Возвращает общее количество изображений в датасете.
        
        Returns:
            int: Количество изображений.
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Загружает и обрабатывает изображение по индексу.
        
        Args:
            idx (int): Индекс изображения в списке.
        
        Returns:
            torch.Tensor: Изображение в виде тензора размером [1, H, W] с нормализацией в диапазоне [0, 1].
        
        Если происходит ошибка при чтении изображения, возвращается тензор заполненный нулями.
        """
        try:
            img_path = self.image_files[idx]
            img = Image.open(img_path).convert('L')
            img = img.resize(self.target_size, Image.BICUBIC)
            img = np.array(img).astype(np.float32) / 255.0
            if self.transform:
                img = self.transform(img)
            img = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]
            return img
        except Exception as e:
            print(f"Ошибка при чтении изображения {self.image_files[idx]}: {e}")
            return torch.zeros(1, *self.target_size)


class ComCNN(nn.Module):
    """
    Сверточная нейронная сеть для сжатия изображений (ComCNN).
    
    Архитектура сети включает несколько слоев свертки, нормализации и функции активации ReLU.
    Глубина сети увеличена, а также применяется BatchNorm для улучшения сходимости.
    
    Атрибуты:
        layer1 (nn.Sequential): Первый сверточный слой с BatchNorm и ReLU.
        layer2 (nn.Sequential): Второй слой с downsample (stride=2), BatchNorm и ReLU.
        layer3 (nn.Sequential): Третий сверточный слой с BatchNorm и ReLU.
        layer4 (nn.Sequential): Последний слой, приводящий к выходу с одним каналом.
    
    Методы:
        forward(x): Выполняет прямой проход по сети.
    """
    def __init__(self):
        super(ComCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # downsample
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def forward(self, x):
        """
        Выполняет прямой проход входного тензора через все слои сети.
        
        Args:
            x (torch.Tensor): Входной тензор размера [N, 1, H, W].
        
        Returns:
            torch.Tensor: Выход сети, представляющий сжатое изображение.
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ResidualBlock(nn.Module):
    """
    Реализует простой residual-блок, используемый в сети RecCNN.
    
    Блок состоит из двух сверточных слоев с последующей нормализацией и нелинейной активацией.
    После второго сверточного слоя происходит сложение с исходным входом (skip connection),
    а затем применяется функция активации ReLU.
    
    Атрибуты:
        conv1 (nn.Conv2d): Первый сверточный слой.
        bn1 (nn.BatchNorm2d): BatchNorm после первого сверточного слоя.
        conv2 (nn.Conv2d): Второй сверточный слой.
        bn2 (nn.BatchNorm2d): BatchNorm после второго сверточного слоя.
    
    Методы:
        forward(x): Пропускает вход через блок и возвращает результат.
    """
    def __init__(self, channels=64):
        """
        Инициализирует residual-блок.
        
        Args:
            channels (int, optional): Количество каналов входного и выходного тензора. По умолчанию 64.
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        """
        Выполняет прямой проход через residual-блок.
        
        Args:
            x (torch.Tensor): Входной тензор.
        
        Returns:
            torch.Tensor: Результат после применения сверточных операций, BatchNorm, ReLU и сложения с исходным входом.
        """
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class RecCNN(nn.Module):
    """
    Нейронная сеть для реконструкции изображений (RecCNN), использующая residual-блоки.
    
    Архитектура сети включает:
      - Начальный сверточный слой с BatchNorm и ReLU.
      - Последовательность residual-блоков для глубокого извлечения признаков.
      - Завершающий сверточный слой, приводящий к восстановленному изображению.
    
    Атрибуты:
        conv1 (nn.Sequential): Начальный сверточный слой.
        resblocks (nn.Sequential): Последовательность residual-блоков.
        conv_last (nn.Conv2d): Завершающий сверточный слой.
    
    Методы:
        forward(x): Выполняет прямой проход по сети.
    """
    def __init__(self, num_resblocks=3):
        """
        Инициализирует сеть RecCNN.
        
        Args:
            num_resblocks (int, optional): Количество residual-блоков, используемых в сети. По умолчанию 3.
        """
        super(RecCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Последовательность residual-блоков
        self.resblocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_resblocks)])
        # Завершающий сверточный слой
        self.conv_last = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        """
        Выполняет прямой проход входного тензора через сеть.
        
        Args:
            x (torch.Tensor): Входной тензор размера [N, 1, H, W].
        
        Returns:
            torch.Tensor: Восстановленное изображение.
        """
        x = self.conv1(x)
        x = self.resblocks(x)
        x = self.conv_last(x)
        return x


def approx_codec(x, block_size=2):
    """
    Имитация работы кодека с блочным шумом, который добавляет артефакты квантования.
    
    Функция делит изображение (тензор) на блоки заданного размера и в каждом блоке добавляет небольшой случайный шум.
    Это позволяет смоделировать блочные артефакты, схожие с теми, что возникают при JPEG-сжатии.
    
    Args:
        x (torch.Tensor): Входной тензор размера [N, 1, H, W] с значениями в диапазоне [0, 1].
        block_size (int, optional): Размер блока, по умолчанию 2.
    
    Returns:
        torch.Tensor: Тензор с добавленным шумом, той же размерности, что и x.
    
    Если размеры H или W не кратны block_size, можно добавить паддинг (не реализовано в данной версии).
    В случае возникновения исключения, функция выводит сообщение об ошибке и возвращает исходный тензор.
    """
    try:
        # x: [N, C=1, H, W]
        n, c, h, w = x.shape
        # Преобразование x в форму [N, H, W] (так как C=1)
        x_2d = x.view(n, h, w)

        # Определяем количество блоков по высоте и ширине
        blocks_h = h // block_size
        blocks_w = w // block_size

        x_copy = x_2d.clone()

        # Обходим все блоки и добавляем небольшой шум
        for i in range(blocks_h):
            for j in range(blocks_w):
                start_h = i * block_size
                start_w = j * block_size
                end_h = start_h + block_size
                end_w = start_w + block_size

                # Создание шума для текущего блока
                block_noise = (torch.rand(1, device=x.device) - 0.5) / 100.0
                x_copy[:, start_h:end_h, start_w:end_w] += block_noise

        x_copy = torch.clamp(x_copy, 0.0, 1.0)
        # Возвращаем тензор в исходной форме [N, C, H, W]
        return x_copy.view(n, c, h, w)
    except Exception as e:
        print(f"Ошибка в approx_codec: {e}")
        return x


def combined_loss(output, target, alpha=0.5):
    """
    Вычисляет комбинированную функцию потерь, объединяющую MSE и MS-SSIM.
    
    Потери рассчитываются как взвешенная сумма:
        alpha * MSE(output, target) + (1 - alpha) * (1 - MS-SSIM(output, target))
    
    Args:
        output (torch.Tensor): Выход модели.
        target (torch.Tensor): Целевое (исходное) изображение.
        alpha (float, optional): Вес для MSE. (1 - alpha) используется для (1 - MS-SSIM). По умолчанию 0.5.
    
    Returns:
        torch.Tensor: Скаляром представленный результат комбинированной потери.
    
    MS-SSIM вычисляется с использованием библиотеки pytorch_msssim.
    """
    mse = F.mse_loss(output, target)
    msssim_val = ms_ssim(output, target, data_range=1.0)
    # MS-SSIM находится в диапазоне [0,1], где 1 – идеальное соответствие
    return alpha * mse + (1 - alpha) * (1 - msssim_val)


def train_model(train_dataset, val_dataset, num_epochs=50, batch_size=16, learning_rate=1e-3, device='cuda'):
    """
    Обучает модели ComCNN и RecCNN на обучающем датасете с последующей валидацией.
    
    Обучение разделено на две фазы:
      1. Обучение сети RecCNN при фиксированном ComCNN.
      2. Обучение сети ComCNN при фиксированном RecCNN.
    
    Для каждой фазы вычисляется основная ошибка (комбинированная MSE + MS-SSIM) и 
    ранговая потеря, сравнивающая ошибку модели с ошибкой JPEG-сжатия.
    
    Args:
        train_dataset (Dataset): Обучающий датасет.
        val_dataset (Dataset): Валидационный датасет.
        num_epochs (int, optional): Количество эпох обучения. По умолчанию 50.
        batch_size (int, optional): Размер батча. По умолчанию 16.
        learning_rate (float, optional): Скорость обучения для оптимизаторов. По умолчанию 1e-3.
        device (str, optional): Устройство для вычислений ('cuda' или 'cpu'). По умолчанию 'cuda'.
    
    Returns:
        tuple: Обученные модели (com_cnn, rec_cnn).
    
    Детали реализации:
        - Для оптимизации используются Adam-оптимизаторы и ReduceLROnPlateau для адаптивного изменения скорости обучения.
        - Для каждого батча дополнительно вычисляется ранговая потеря, сравнивающая ошибку модели с ошибкой JPEG-сжатия.
        - Лучшие модели (наименьшая валидационная потеря) сохраняются на диск.
        - В ходе обучения сохраняются изображения реконструкции и график изменения потерь.
    
    Внутренняя функция:
        compute_jpeg_tensor(image_tensor, quality=75):
            Преобразует изображение-тензор в JPEG-сжатый тензор.
    """
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    # Гиперпараметры для ранговой потери
    ranking_loss_weight = 1.0
    margin = 0.0  # Порог для ранговой потери

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    com_cnn = ComCNN().to(device)
    rec_cnn = RecCNN(num_resblocks=3).to(device)

    optimizer_com = optim.Adam(com_cnn.parameters(), lr=learning_rate)
    optimizer_rec = optim.Adam(rec_cnn.parameters(), lr=learning_rate)

    scheduler_com = ReduceLROnPlateau(optimizer_com, mode='min', factor=0.5, patience=5, verbose=True)
    scheduler_rec = ReduceLROnPlateau(optimizer_rec, mode='min', factor=0.5, patience=5, verbose=True)

    train_losses = []
    val_losses = []

    # Папка для сохранения изображений эпох
    output_dir = "epoch_images"
    os.makedirs(output_dir, exist_ok=True)

    best_val_loss = float('inf')

    def compute_jpeg_tensor(image_tensor, quality=75):
        """
        Преобразует изображение-тензор в JPEG-сжатый тензор.
        
        Args:
            image_tensor (torch.Tensor): Тензор изображения размера [1, H, W] с значениями в диапазоне [0, 1].
            quality (int, optional): Параметр качества JPEG-сжатия (1-100). По умолчанию 75.
        
        Returns:
            torch.Tensor: JPEG-сжатое изображение в виде тензора размера [1, 1, H, W].
        
        Процесс:
            - Преобразование тензора в NumPy-массив с масштабированием до [0, 255].
            - Создание изображения с помощью PIL и изменение его размера до 256x256 с использованием bicubic.
            - Сохранение изображения в in-memory буфер в формате JPEG.
            - Чтение изображения из буфера, нормализация и преобразование обратно в тензор.
        """
        image_np = image_tensor.cpu().numpy().squeeze() * 255.0
        image_pil = Image.fromarray(image_np.astype(np.uint8))
        image_pil = image_pil.resize((256, 256), Image.BICUBIC).convert('L')

        buffer = BytesIO()
        image_pil.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        jpeg_pil = Image.open(buffer).convert('L')
        jpeg_np = np.array(jpeg_pil).astype(np.float32) / 255.0
        jpeg_tensor = torch.from_numpy(jpeg_np).unsqueeze(0).unsqueeze(0).to(device)
        return jpeg_tensor

    for epoch in range(num_epochs):
        com_cnn.train()
        rec_cnn.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for batch in progress_bar:
            batch = batch.to(device)

            # -----------------------------
            # Фаза 1: Обучение RecCNN при фиксированном ComCNN
            # -----------------------------
            com_cnn.eval()
            rec_cnn.train()
            optimizer_rec.zero_grad()

            compact = com_cnn(batch)
            coded = approx_codec(compact, block_size=8)
            residual = rec_cnn(coded)
            output = coded - residual
            output_upsampled = F.interpolate(output, size=batch.shape[2:], mode='bicubic', align_corners=False)
            # Вычисление основной ошибки для батча
            loss_rec = combined_loss(output_upsampled, batch, alpha=0.5)

            # Вычисление ранговой потери по сравнению с JPEG для каждого изображения в батче
            batch_ranking_loss = 0.0
            for i in range(batch.size(0)):
                loss_model = combined_loss(output_upsampled[i].unsqueeze(0), batch[i].unsqueeze(0), alpha=0.5)
                jpeg_tensor = compute_jpeg_tensor(batch[i].unsqueeze(0))
                loss_jpeg = combined_loss(jpeg_tensor, batch[i].unsqueeze(0), alpha=0.5)
                batch_ranking_loss += torch.relu(loss_model - loss_jpeg + margin)
            batch_ranking_loss /= batch.size(0)

            total_loss_rec = loss_rec + ranking_loss_weight * batch_ranking_loss
            total_loss_rec.backward()
            optimizer_rec.step()

            # -----------------------------
            # Фаза 2: Обучение ComCNN при фиксированном RecCNN
            # -----------------------------
            rec_cnn.eval()
            com_cnn.train()
            optimizer_com.zero_grad()

            compact = com_cnn(batch)
            coded = approx_codec(compact, block_size=8)
            residual = rec_cnn(coded)
            output = coded - residual
            output_upsampled = F.interpolate(output, size=batch.shape[2:], mode='bicubic', align_corners=False)
            loss_com = combined_loss(output_upsampled, batch, alpha=0.5)

            batch_ranking_loss = 0.0
            for i in range(batch.size(0)):
                loss_model = combined_loss(output_upsampled[i].unsqueeze(0), batch[i].unsqueeze(0), alpha=0.5)
                jpeg_tensor = compute_jpeg_tensor(batch[i].unsqueeze(0))
                loss_jpeg = combined_loss(jpeg_tensor, batch[i].unsqueeze(0), alpha=0.5)
                batch_ranking_loss += torch.relu(loss_model - loss_jpeg + margin)
            batch_ranking_loss /= batch.size(0)

            total_loss_com = loss_com + ranking_loss_weight * batch_ranking_loss
            total_loss_com.backward()
            optimizer_com.step()

            running_loss += (total_loss_rec.item() + total_loss_com.item())
            progress_bar.set_postfix(loss=running_loss/(progress_bar.n+1))

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

        # -----------------------------
        # Валидация с визуализацией и сравнением с JPEG
        # -----------------------------
        com_cnn.eval()
        rec_cnn.eval()
        val_loss = 0.0

        random_batch = random.choice(list(val_loader))

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                compact = com_cnn(batch)
                coded = approx_codec(compact, block_size=8)
                residual = rec_cnn(coded)
                output = coded - residual
                output_upsampled = F.interpolate(output, size=batch.shape[2:], mode='bicubic', align_corners=False)
                loss = combined_loss(output_upsampled, batch, alpha=0.5)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")

        # Визуализация для одного примера из валидационного набора
        sample_batch = random_batch.to(device)
        with torch.no_grad():
            compact = com_cnn(sample_batch)
            coded = approx_codec(compact, block_size=8)
            residual = rec_cnn(coded)
            output = coded - residual
            output_upsampled = F.interpolate(output, size=sample_batch.shape[2:], mode='bicubic', align_corners=False)

            # JPEG-сжатие для сравнения
            sample_np = sample_batch[0].cpu().numpy().squeeze()
            original_pil = Image.fromarray((sample_np * 255).astype(np.uint8))
            # Изменение размера и преобразование для JPEG
            original_pil = original_pil.resize((256, 256), Image.BICUBIC).convert('L')
            buffer = BytesIO()
            original_pil.save(buffer, format='JPEG', quality=75)
            buffer.seek(0)
            jpeg_pil = Image.open(buffer).convert('L')
            jpeg_np = np.array(jpeg_pil).astype(np.float32) / 255.0
            jpeg_tensor = torch.from_numpy(jpeg_np).unsqueeze(0).unsqueeze(0).to(device)

            nn_loss = combined_loss(output_upsampled, sample_batch, alpha=0.5)
            jpeg_loss = combined_loss(jpeg_tensor, sample_batch[0].unsqueeze(0), alpha=0.5)

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(sample_np, cmap='gray', vmin=0, vmax=1)
        plt.title("Оригинал")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        nn_img = output_upsampled[0].cpu().numpy().squeeze()
        plt.imshow(nn_img, cmap='gray', vmin=0, vmax=1)
        plt.title(f"Нейросеть\nLoss: {nn_loss.item():.4f}")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(jpeg_np, cmap='gray', vmin=0, vmax=1)
        plt.title(f"JPEG (quality=75)\nLoss: {jpeg_loss.item():.4f}")
        plt.axis('off')

        plt.savefig(os.path.join(output_dir, f"reconstruction_epoch_{epoch+1}.png"))
        plt.close()

        # Сохранение моделей, если валидационная потеря улучшилась
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(com_cnn.state_dict(), "best_ComCNN_trained.pth")
            torch.save(rec_cnn.state_dict(), "best_RecCNN_trained.pth")
            print(f"Новый лучший результат на эпохе {epoch+1} (Validation Loss: {avg_val_loss:.4f}). Модели сохранены.")
        else:
            print(f"Эпоха {epoch+1}: Validation Loss не улучшился (Best: {best_val_loss:.4f}). Модели не сохранены.")

    # Построение графика изменения потерь
    plt.figure(figsize=(8,6))
    plt.plot(range(1, num_epochs+1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs+1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE + MS-SSIM + Ranking)")
    plt.title("Loss Curves")
    plt.legend()
    plt.savefig("loss_curves.png")
    plt.show()

    return com_cnn, rec_cnn


def main():
    """
    Основная функция для запуска процесса обучения моделей.
    
    Последовательность действий:
      1. Определяется каталог с данными (dataset_dir) и собираются пути к изображениям с помощью get_image_paths.
      2. Пути к изображениям перемешиваются и разделяются на обучающую (80%) и валидационную (20%) выборки.
      3. Создаются объекты ImageDataset для обучающей и валидационной выборок с заданными трансформациями и размером (256x256).
      4. Определяется устройство для вычислений (GPU, если доступно, иначе CPU).
      5. Вызывается функция train_model для обучения моделей.
    
    В случае возникновения ошибок (например, отсутствие изображений в каталоге) выводится сообщение об ошибке.
    """
    try:
        dataset_dir = "DIV2K_train_LR_bicubic"
        all_paths = get_image_paths(dataset_dir)
        if not all_paths:
            raise ValueError("Не найдено изображений в указанном каталоге!")

        random.shuffle(all_paths)
        n_total = len(all_paths)
        n_train = int(0.8 * n_total)
        train_paths = all_paths[:n_train]
        val_paths = all_paths[n_train:]

        print(f"Всего изображений: {n_total}, Обучающих: {len(train_paths)}, Валидационных: {len(val_paths)}")

        train_dataset = ImageDataset(train_paths, transform=augment_image, target_size=(256, 256))
        val_dataset = ImageDataset(val_paths, transform=None, target_size=(256, 256))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Используем устройство: {device}")

        com_model, rec_model = train_model(train_dataset, val_dataset,
                                           num_epochs=10,
                                           batch_size=6,
                                           learning_rate=1e-3,
                                           device=device)
    except Exception as e:
        print(f"Ошибка в функции main: {e}")


if __name__ == "__main__":
    main()
