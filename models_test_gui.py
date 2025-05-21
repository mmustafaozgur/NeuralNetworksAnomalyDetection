import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from torchvision import transforms
import os

# —————— Lightning import'ları ——————
from anomalib.models.image.fastflow.lightning_model import Fastflow as FastflowLightning
from anomalib.models.image.padim.lightning_model    import Padim
from anomalib.models.image.cfa.lightning_model      import Cfa

def load_model(model_name: str, checkpoint_path: str, device: torch.device):
    m = model_name.lower()
    if m == "fastflow":
        model = FastflowLightning.load_from_checkpoint(checkpoint_path, map_location=device)
    elif m == "padim":
        model = Padim.load_from_checkpoint(checkpoint_path, map_location=device)
    elif m == "cfa":
        model = Cfa.load_from_checkpoint(checkpoint_path, map_location=device)
    else:
        raise ValueError(f"Desteklenmeyen model: {model_name}")
    return model.to(device).eval()

class AnomalyApp:
    def __init__(self, root):
        self.root = root
        root.title("Anomali Tespiti")
        root.geometry("800x600")
        root.resizable(True, True)
        root.configure(bg="#f0f0f0")
        
        # Tema ve stiller
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Renk paleti
        self.primary_color = "#3498db"  # Mavi
        self.secondary_color = "#2c3e50"  # Koyu mavi
        self.accent_color = "#e74c3c"  # Kırmızı
        self.bg_color = "#f0f0f0"  # Açık gri
        self.text_color = "#2c3e50"  # Koyu mavi
        
        # Buton stilleri
        self.style.configure('Primary.TButton', 
                             background=self.primary_color, 
                             foreground='white', 
                             font=('Helvetica', 10, 'bold'),
                             borderwidth=0,
                             padding=5)
        
        self.style.configure('Secondary.TButton', 
                             background=self.secondary_color, 
                             foreground='white',
                             font=('Helvetica', 10),
                             borderwidth=0,
                             padding=5)
        
        self.style.map('Primary.TButton', 
                       background=[('active', '#2980b9')])
        
        self.style.map('Secondary.TButton', 
                       background=[('active', '#1f2c38')])
        
        # Label stilleri
        self.style.configure('TLabel', 
                             background=self.bg_color, 
                             foreground=self.text_color,
                             font=('Helvetica', 10))
        
        self.style.configure('Header.TLabel', 
                             background=self.bg_color, 
                             foreground=self.text_color,
                             font=('Helvetica', 14, 'bold'))
        
        # Combobox stilleri
        self.style.configure('TCombobox', 
                             background=self.bg_color, 
                             fieldbackground='white',
                             padding=5)
        
        # Entry stilleri
        self.style.configure('TEntry', 
                             padding=5)
        
        # Değişkenler
        self.image_path = tk.StringVar()
        self.model_choice = tk.StringVar(value="fastflow")
        self.status_text = tk.StringVar(value="Hazır")
        
        # Ana çerçeve
        main_frame = ttk.Frame(root, padding="20", style='TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Başlık
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(header_frame, text="Anomali Tespiti Aracı", style='Header.TLabel').pack()
        
        # Ayarlar çerçevesi
        settings_frame = ttk.LabelFrame(main_frame, text="Ayarlar", padding="10")
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Model seçimi
        model_frame = ttk.Frame(settings_frame)
        model_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(model_frame, text="Model:").pack(side=tk.LEFT, padx=(0, 10))
        model_menu = ttk.Combobox(model_frame, textvariable=self.model_choice, 
                                values=["fastflow", "padim", "cfa"],
                                state="readonly", width=15)
        model_menu.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Görsel seçimi
        image_frame = ttk.Frame(settings_frame)
        image_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(image_frame, text="Görsel:").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Entry(image_frame, textvariable=self.image_path).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        ttk.Button(image_frame, text="Görsel Seç", command=self.browse_image, style='Secondary.TButton').pack(side=tk.RIGHT)
        
        # Çalıştırma düğmesi
        run_frame = ttk.Frame(settings_frame)
        run_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(run_frame, text="ANOMALİ TESPİTİ YAP", command=self.run_inference, style='Primary.TButton').pack(fill=tk.X)
        
        # Durum çubuğu
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(status_frame, text="Durum:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(status_frame, textvariable=self.status_text).pack(side=tk.LEFT)
        
        # Sonuç gösterimi için frame
        self.result_frame = ttk.Frame(main_frame)
        self.result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Görsel önizleme
        self.preview_frame = ttk.LabelFrame(self.result_frame, text="Görsel Önizleme")
        self.preview_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.preview_label = ttk.Label(self.preview_frame)
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # İlk görsel yükleme
        self.load_placeholder_image()
        
        # Sonuç gösterimi
        self.result_display_frame = ttk.LabelFrame(self.result_frame, text="Anomali Haritası")
        self.result_display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Model ve cihaz ayarları
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.status_text.set(f"Hazır (Cihaz: {self.device})")
        
        self.model_paths = {
            "fastflow": "results/fastflow/wood/v0/weights/lightning/model.ckpt",
            "padim":    "results/padim/wood/v0/weights/lightning/model.ckpt",
            "cfa":      "results/cfa/wood/v0/weights/lightning/model.ckpt"
        }
        self.loaded_models = {}

        # Manuel pipeline: PIL→256×256 Resize→ToTensor→Normalize
        self.manual_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std= [0.229, 0.224, 0.225]
            ),
        ])
    
    def load_placeholder_image(self):
        # Varsayılan bir görsel göster
        placeholder = Image.new('RGB', (256, 256), color='#f0f0f0')
        placeholder_tk = ImageTk.PhotoImage(placeholder)
        self.preview_label.configure(image=placeholder_tk)
        self.preview_label.image = placeholder_tk  # Referansı korumak için

    def browse_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp"),("All Files","*.*")]
        )
        if path:
            self.image_path.set(path)
            self.update_preview(path)
    
    def update_preview(self, path):
        # Görsel önizleme güncelleme
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((256, 256), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            self.preview_label.configure(image=img_tk)
            self.preview_label.image = img_tk  # Referansı korumak için
        except Exception as e:
            messagebox.showerror("Hata", f"Görsel yüklenemedi:\n{e}")

    def run_inference(self):
        img_path = self.image_path.get()
        model_key = self.model_choice.get().lower()
        
        if not img_path:
            messagebox.showwarning("Uyarı", "Önce bir görsel seçin.")
            return
            
        if not os.path.exists(img_path):
            messagebox.showerror("Hata", "Belirtilen dosya bulunamadı.")
            return

        # Durum güncelleme
        self.status_text.set("Model yükleniyor...")
        self.root.update()

        # Model yükleme
        if model_key not in self.loaded_models:
            try:
                self.status_text.set(f"{model_key.upper()} modeli yükleniyor...")
                self.root.update()
                
                self.loaded_models[model_key] = load_model(
                    model_key,
                    self.model_paths[model_key],
                    self.device
                )
            except Exception as e:
                messagebox.showerror("Hata", f"Model yüklenemedi:\n{e}")
                self.status_text.set("Hata: Model yüklenemedi")
                return

        model = self.loaded_models[model_key]
        
        # Durum güncelleme
        self.status_text.set("Anomali tespiti yapılıyor...")
        self.root.update()

        try:
            # Orijinal ve ön işlenmiş görsel
            orig = Image.open(img_path).convert("RGB")
            
            # Manuel ön-işlem → batch → device
            tensor = self.manual_transform(orig).unsqueeze(0).to(self.device)

            # İnference
            with torch.no_grad():
                out = model(tensor)
                if isinstance(out, dict):
                    am_t = out.get("anomaly_map") or out.get("anomaly_maps")
                elif hasattr(out, "anomaly_map"):
                    am_t = out.anomaly_map
                else:
                    am_t = out
                am = am_t.squeeze().cpu().numpy()

            # Min–max normalize anomaly map
            am = (am - am.min())/(am.max()-am.min()+1e-6)
            
            # Matplotlib figürü oluştur
            for widget in self.result_display_frame.winfo_children():
                widget.destroy()
                
            fig, ax = plt.subplots(figsize=(5, 5), dpi=80)
            im = ax.imshow(am, cmap="jet", vmin=0, vmax=1)
            ax.set_title("Anomali Haritası")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            
            # Figürü Tkinter'e ekle
            canvas = FigureCanvasTkAgg(fig, master=self.result_display_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Durum güncelleme
            self.status_text.set(f"Analiz tamamlandı. Model: {model_key.upper()}")
            
        except Exception as e:
            messagebox.showerror("Hata", f"İşlem sırasında bir hata oluştu:\n{e}")
            self.status_text.set("Hata oluştu")

if __name__ == "__main__":
    root = tk.Tk()
    app = AnomalyApp(root)
    root.mainloop()