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

Kurulum
Gereksinimler

Python 3.12+
PyTorch 1.10+
CUDA 11.1+ (GPU kullanımı için)
Anomalib 2.0.0

Conda İle Sanal Ortam Kurulumu

Repository'yi klonlayın:

```bash
git clone https://github.com/mmustafaozgur/NeuralNetworksAnomalyDetection.git
cd NeuralNetworksAnomalyDetection
```

Conda kullanarak sanal ortam oluşturun:

```bash
conda create -n anomaly-env python=3.8
conda activate anomaly-env
```

PyTorch ve CUDA kurulumu:

```bash
# CUDA 11.6 için
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
```

# CPU kullanımı için
# conda install pytorch torchvision torchaudio cpuonly -c pytorch

Gerekli paketleri yükleyin:

```bash
pip install -r requirements.txt
```

Anomalib kütüphanesini yükleyin:
```bash
pip install anomalib[full]
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


## Proje Klasör Yapısı

```
ahsap-anomali-tespiti/
│
├── cfa.py                    # CFA modeli eğitim ve test kodu
├── fastflow_padim.py         # FastFlow ve PADIM modelleri için kod
├── models_test_gui.py        # Masaüstü uygulaması
│
├── requirements.txt          # Gerekli paketler
└── README.md                 
```






---

Bu proje, Eskişehir Osmangazi Üniversitesi Bilgisayar Mühendisliği Bölümü Neural Networks dersi kapsamında geliştirilmiştir.
