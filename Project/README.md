🎵 Spotify Hit Predictor

Makine öğrenmesi kullanarak bir şarkının popüler olup olmayacağını tahmin eden bir Python projesidir.
Spotify’ın 2020 “TidyTuesday” veri seti kullanılarak analiz, görselleştirme ve modelleme yapılmıştır.

📂 Proje İçeriği

Bu proje üç ana bileşenden oluşur:

1) Veri Analizi

Açıklayıcı istatistikler (describe)

Korelasyon matrisi

Isı haritası (correlation heatmap)

Pairplot analizleri

2) Özellik Mühendisliği

Eksik veri temizleme

Kategorik → numerik dönüşüm

Ölçeklendirme işlemleri (StandardScaler)

Özellik seçimi (feature selection)

3) Makine Öğrenmesi Modelleri

Aşağıdaki modellerle popülerlik tahmini yapılmıştır:

Linear Regression

Random Forest Regressor

Ayrıca:

RMSE hesaplaması

Gerçek vs tahmin karşılaştırması

Feature importance grafikleri

📊 Kullanılan Veri Seti

Dataset:
Spotify Songs - 2020
Kaynak: TidyTuesday
Ham veri linki:
https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv

Veri seti toplam 32.833 şarkı ve 23 özellik içerir.

🚀 Nasıl Çalıştırılır?
1. Depoyu klonla
```git clone https://github.com/kullanici_adi/Spotify-Hit-Predictor.git
cd Spotify-Hit-Predictor```

2. Gerekli kütüphaneleri kur
```pip install -r requirements.txt```

3. Modeli çalıştır
```python3 hit_predictor.py --data data/spotify_2020.csv```

Oluşturulan çıktılar:
outputs/
 ├─ dataset_describe.csv
 ├─ correlation_matrix.csv
 └─ plots/
       ├─ correlation_matrix.png
       ├─ pairplot_subset.png
       └─ feature_importance.png

🧠 Model Performansı (Özet)

Linear Regression → Basit temel model

Random Forest → Daha yüksek doğruluk, daha düşük RMSE

En önemli özellikler:

Instrumentalness

Duration (ms)

Loudness

Energy

Tempo

Bir şarkının hit olmasında tek bir güçlü korelasyon yoktur, fakat yukarıdaki özellikler toplu olarak iyi sinyal sağlar.

📌 Projenin Amacı

Bu proje:

Spotify verilerini analiz etmek

Hit şarkıları etkileyen faktörleri çıkarmak

Basit bir tahmin modeli geliştirmek

Makine öğrenmesi eğitimlerinde örnek proje sağlamak
amaçlarıyla hazırlanmıştır.

📁 Dosya Yapısı
Spotify-Hit-Predictor/
│── data/
│     └─ spotify_2020.csv
│── outputs/
│     └─ plots/
│── hit_predictor.py
│── requirements.txt
│── README.md

🎯 Neler Öğrenebilirsin?

Bu projeyle:

Veri analizi (Pandas, Seaborn)

Olasılıksal tahminler

Feature engineering

Makine öğrenmesi modelli oluşturma

Model değerlendirme (RMSE, R²)

Grafik üretimi
gibi temel ML becerilerini öğrenirsin.

🤝 Katkı Yapmak

Pull request gönderebilirsin.
Yeni model, preprocessing veya görselleştirme eklemek isteyen herkes katkıda bulunabilir.

📜 Lisans

MIT Lisansı altında yayınlanmıştır.
