# Ahşap Yüzeylerde Anomali Tespiti ve Segmentasyonu

Bu proje, ahşap yüzeylerindeki anomalileri tespit etmek ve segmente etmek için denetimsiz öğrenme yaklaşımını kullanan derin öğrenme modellerini içermektedir. Üç farklı state-of-the-art model (FastFlow, PADIM ve CFA) uygulanmış ve performansları karşılaştırılmıştır.

## İçindekiler

- [Genel Bakış](#genel-bakış)
- [Kurulum](#kurulum)
  - [Gereksinimler](#gereksinimler)
  - [Kurulum Adımları](#kurulum-adımları)
- [Veri Seti](#veri-seti)
- [Modelleri Çalıştırma](#modelleri-çalıştırma)
  - [FastFlow ve PADIM Modelleri](#fastflow-ve-padim-modelleri)
  - [CFA Modeli](#cfa-modeli)
- [Masaüstü Uygulaması](#masaüstü-uygulaması)
- [Sonuçlar](#sonuçlar)
- [Katkıda Bulunanlar](#katkıda-bulunanlar)

## Genel Bakış

Bu proje, ahşap yüzeylerinde kusurları tespit etmek için denetimsiz öğrenme yaklaşımını kullanmaktadır. Sadece normal (kusursuz) örnekler kullanılarak eğitilen modeller, test aşamasında anormal (kusurlu) örnekleri tespit etmektedir. 

Projede üç farklı model uygulanmıştır:
- **FastFlow**: Normalizing flow tabanlı anomali tespit modeli
- **PADIM**: Mahalanobis mesafesi tabanlı anomali tespit modeli
- **CFA**: Karşıtsal öğrenme tabanlı anomali tespit modeli

## Kurulum

### Gereksinimler

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.1+ (GPU kullanımı için)
- Anomalib 0.4.0+

### Kurulum Adımları

1. Repository'yi klonlayın:
```bash
git clone https://github.com/yourusername/ahsap-anomali-tespiti.git
cd ahsap-anomali-tespiti
```

2. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

3. Anomalib kütüphanesini yükleyin:
```bash
pip install anomalib
```

## Veri Seti

Proje, MVTec Anomaly Detection veri setinin Wood alt kümesini kullanmaktadır. Veri setini aşağıdaki yapıda organize etmeniz gerekmektedir:

```
wood_dataset/
│
├── wood/
│ ├── train/ 
│ │ └── good/ 
│ │
│ ├── test/ 
│ │ ├── good/ 
│ │ └── defect/ 
│ │
│ └── ground_truth/ 
│ └── defect/
```

MVTec AD veri setini [buradan](https://www.mvtec.com/company/research/datasets/mvtec-ad) indirebilirsiniz.

## Modelleri Çalıştırma

### FastFlow ve PADIM Modelleri

FastFlow ve PADIM modellerini eğitmek ve test etmek için aşağıdaki komutu kullanın:

```bash
python fastflow_padim.py --model [fastflow|padim] --data-path wood_dataset/wood --max-time 20
```

Parametreler:
- `--model`: Kullanılacak model ("fastflow" veya "padim")
- `--data-path`: Veri setinin konumu
- `--max-time`: Maksimum eğitim süresi (dakika cinsinden)

### CFA Modeli

CFA modelini eğitmek ve test etmek için:

```bash
python cfa.py --model cfa --data-path wood_dataset/wood --max-time 20
```

Parametreler:
- `--model`: Model adı ("cfa")
- `--data-path`: Veri setinin konumu
- `--max-time`: Maksimum eğitim süresi (dakika cinsinden)

## Masaüstü Uygulaması

Eğitilen modelleri test etmek için geliştirilen masaüstü uygulamasını aşağıdaki komutla çalıştırabilirsiniz:

```bash
python models_test_gui.py
```

Uygulama, aşağıdaki özelliklere sahiptir:
- Model seçimi (FastFlow, PADIM, CFA)
- Görüntü seçimi ve önizleme
- Anomali haritası görselleştirme

**Not**: Uygulamanın çalışması için `results` klasöründe eğitilmiş model dosyalarının bulunması gerekmektedir. Modelleri önceden eğittiğinizden emin olun.

## Sonuçlar

Projede kullanılan modellerin performans metrikleri:

| Model | Görüntü AUROC | Görüntü F1 Score | Piksel AUROC | Piksel F1 Score |
|-------|--------------|-----------------|-------------|----------------|
| CFA   | 0.8091       | 0.6827          | 0.9386      | 0.0239         |
| FastFlow | 0.7425    | 0.7119          | 0.9023      | 0.2690         |
| PADIM | 0.8258       | 0.7973          | 0.9318      | 0.2931         |

## Hata Giderme ve Sık Karşılaşılan Sorunlar

### CUDA Bellek Hatası

Eğer "CUDA out of memory" hatası alırsanız, batch size değerini düşürmeyi deneyin. Bu, `fastflow_padim.py` ve `cfa.py` dosyalarında `create_datamodule` fonksiyonu içinde ayarlanabilir.

### Anomalib Sürüm Uyumsuzluğu

Anomalib kütüphanesi hızlı geliştiği için, sürüm uyumsuzluğu yaşayabilirsiniz. Uyumsuzluk durumunda, kodun çalıştığı sürümü yüklemek için:

```bash
pip install anomalib==0.4.0
```

### PyTorch Lightning Hatası

PyTorch Lightning'in yeni sürümlerinde bazı değişiklikler oldu. Kodumuz hem eski hem de yeni sürümlerle uyumlu olacak şekilde tasarlanmıştır. Ancak bir hata alırsanız, belirli bir sürümü yüklemek sorunları çözebilir:

```bash
pip install lightning==2.0.0
```

## Proje Klasör Yapısı

```
ahsap-anomali-tespiti/
│
├── cfa.py                    # CFA modeli eğitim ve test kodu
├── fastflow_padim.py         # FastFlow ve PADIM modelleri için kod
├── models_test_gui.py        # Masaüstü uygulaması
│
├── results/                  # Eğitim sonuçları
│   ├── cfa/
│   ├── fastflow/
│   └── padim/
│
├── requirements.txt          # Gerekli paketler
└── README.md                 # Bu belge
```

## Katkıda Bulunanlar

- Mustafa ÖZGÜR - 121320191002

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır - detaylar için [LICENSE](LICENSE) dosyasına bakın.

## İletişim

Sorularınız için: [Email Adresiniz]

---

Bu proje, Eskişehir Osmangazi Üniversitesi Bilgisayar Mühendisliği Bölümü Neural Networks dersi kapsamında geliştirilmiştir.
