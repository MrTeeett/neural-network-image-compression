import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
from torchvision import transforms
import matplotlib.pyplot as plt
from pytorch_msssim import ms_ssim
from skimage.metrics import structural_similarity as ssim
from main import ComCNN, RecCNN

# -------------------------------------
# Инициализация моделей
# -------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
com_cnn = ComCNN().to(device)
rec_cnn = RecCNN(num_resblocks=3).to(device)
com_cnn.load_state_dict(torch.load("best_ComCNN_trained.pth", map_location=device))
rec_cnn.load_state_dict(torch.load("best_RecCNN_trained.pth", map_location=device))
com_cnn.eval()
rec_cnn.eval()

# Трансформации
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

# Функция вычисления метрик
def calculate_metrics(original, reconstructed):
    original = np.array(original.convert('L')) / 255.0
    reconstructed = np.array(reconstructed.convert('L').resize(original.shape[::-1])) / 255.0

    mse = np.mean((original - reconstructed) ** 2)
    ssim_score = ssim(original, reconstructed, data_range=1.0)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else 100

    return mse, ssim_score, psnr

# Сжатие с помощью нейросети
def compress_decompress_nn(image_path):
    image = Image.open(image_path).convert('L')
    original_size = image.size
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        compressed = com_cnn(image_tensor)
        residual = rec_cnn(compressed)
        reconstructed = compressed - residual

    reconstructed_image = (reconstructed.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    reconstructed_image = Image.fromarray(reconstructed_image).resize(original_size)
    return reconstructed_image

# Сжатие через JPEG
def compress_decompress_jpeg(image_path, quality=50):
    image = Image.open(image_path).convert('L')
    temp_path = image_path.replace(".png", "_jpeg_compressed.jpg")
    image.save(temp_path, "JPEG", quality=quality)
    jpeg_image = Image.open(temp_path)
    jpeg_image.load()  # Принудительная загрузка в память
    os.remove(temp_path)
    return jpeg_image

# Интерфейс с Tkinter
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Compression Comparison")
        self.root.geometry("1000x600")

        self.label = tk.Label(root, text="Выберите изображение для сжатия", font=("Arial", 14))
        self.label.pack(pady=10)

        self.upload_btn = tk.Button(root, text="Загрузить изображение", command=self.load_image)
        self.upload_btn.pack(pady=5)

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=5)

        self.result_label = tk.Label(root, text="", font=("Arial", 12))
        self.result_label.pack(pady=5)

        self.plot_btn = tk.Button(root, text="Сравнить методы", command=self.compare_methods)
        self.plot_btn.pack(pady=10)
        
        # Новая кнопка для преобразования изображения в ЧБ 256x256 и сохранения
        self.save_bw_btn = tk.Button(root, text="Сохранить ЧБ 256x256", command=self.convert_and_save)
        self.save_bw_btn.pack(pady=5)

        self.original_image = None
        self.image_path = None

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.image_path = file_path
            self.original_image = Image.open(file_path).convert('L')
            img = ImageTk.PhotoImage(self.original_image.resize((300, 300)))
            self.image_label.config(image=img)
            self.image_label.image = img
            self.result_label.config(text="Изображение загружено.")

    def compare_methods(self):
        if not self.image_path:
            messagebox.showerror("Ошибка", "Сначала загрузите изображение.")
            return

        # Сжатие нейросетью
        nn_image = compress_decompress_nn(self.image_path)
        nn_image.save("nn_compressed.png")

        # Сжатие JPEG
        jpeg_image = compress_decompress_jpeg(self.image_path)
        jpeg_image.save("jpeg_compressed.png")

        # Вычисление метрик
        mse_nn, ssim_nn, psnr_nn = calculate_metrics(self.original_image, nn_image)
        mse_jpeg, ssim_jpeg, psnr_jpeg = calculate_metrics(self.original_image, jpeg_image)

        # Вывод метрик
        self.result_label.config(text=f"NN - MSE: {mse_nn:.4f}, SSIM: {ssim_nn:.4f}, PSNR: {psnr_nn:.2f} dB\n"
                                     f"JPEG - MSE: {mse_jpeg:.4f}, SSIM: {ssim_jpeg:.4f}, PSNR: {psnr_jpeg:.2f} dB")

        # Визуализация изображений
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(self.original_image, cmap='gray')
        axes[0].set_title("Оригинал")
        axes[0].axis("off")

        axes[1].imshow(nn_image, cmap='gray')
        axes[1].set_title(f"NN (SSIM: {ssim_nn:.4f})")
        axes[1].axis("off")

        axes[2].imshow(jpeg_image, cmap='gray')
        axes[2].set_title(f"JPEG (SSIM: {ssim_jpeg:.4f})")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig("comparison_plot.png")
        plt.show()
    
    def convert_and_save(self):
        if self.original_image is None:
            messagebox.showerror("Ошибка", "Сначала загрузите изображение.")
            return
        
        # Преобразование в ЧБ (оттенки серого) и изменение размера до 256x256
        bw_image = self.original_image.convert('L').resize((256, 256))
        save_path = "bw_256.png"  # можно изменить путь сохранения или добавить диалог сохранения
        bw_image.save(save_path)
        messagebox.showinfo("Успех", f"Изображение сохранено как {save_path}")

# Запуск приложения
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
