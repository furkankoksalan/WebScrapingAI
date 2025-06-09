# Web Scraping + AI Chat Application

> **Streamlit ve LangChain kullanarak geliştirilmiş, gerçek zamanlı streaming destekli web scraping ve AI sohbet uygulaması**

##  Hakkında

Bu proje, web sitelerinden otomatik içerik çıkarma (web scraping) ve yapay zeka destekli sohbet özelliklerini birleştiren gelişmiş bir uygulamadır. Real-time streaming, RAG (Retrieval-Augmented Generation) teknolojisi ve modern chat arayüzü ile Streamlit, LangChain ve OpenAI GPT modelleri kullanılarak geliştirilmiştir.

Temel amacı, kullanıcıların farklı web sitelerinden içerik çıkararak bu bilgiler üzerinde akıllı sorgu ve analiz yapabilmelerini sağlamaktır. Sistem otomatik olarak web içeriğini işler, vektör veritabanında saklar ve doğru cevaplar vermek için konuşma bağlamını korur.

## Ana Özellikler

- ** Çoklu web sitesi scraping**: Birden fazla URL'den eş zamanlı içerik çıkarma
- ** Real-time streaming chat**: Kelime kelime akan modern sohbet deneyimi  
- ** RAG teknolojisi**: Scrape edilen içerikle desteklenmiş AI yanıtları
- ** 5 farklı GPT model desteği**: GPT-4o Mini'den GPT-4o'ya kadar
- ** Sohbet oturumu yönetimi**: Çoklu oturum desteği ve otomatik kaydetme
- ** FAISS vektör veritabanı**: Hızlı ve verimli semantic arama
- ** Otomatik kaynak atıfı**: Cevaplarda hangi kaynak kullanıldığını gösterme
- ** Modern Streamlit UI**: Native chat componentleri ile temiz arayüz
- **⚙ Streaming/Normal mod**: Kullanıcı tercihine göre yanıt görüntüleme

## Hızlı Başlangıç

### Gereksinimler
- Python 3.8+
- OpenAI API anahtarı
- İnternet bağlantısı

### Kurulum

```bash
# Repository'yi klonlayın
git clone https://github.com/username/web-scraping-ai-chat.git
cd web-scraping-ai-chat

# Sanal ortam oluşturun
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt

# Environment dosyası oluşturun
echo "OPENAI_API_KEY=your-openai-api-key-here" > .env

# Uygulamayı çalıştırın
streamlit run main.py
```

Tarayıcınızda `http://localhost:8501` adresine gidin ve kullanmaya başlayın!

## 📋 Kullanım Rehberi

### 1. Model ve Ayarlar
- **Model seçimi**: GPT-4o Mini (ekonomik) → GPT-4o (en gelişmiş)
- **Temperature**: Yaratıcılık seviyesi (0.0-1.0)
- **Max Tokens**: Yanıt uzunluğu (100-4000)
- **Streaming**: Real-time yanıt görüntüleme

### 2. Web Scraping
```
Sidebar → Web Scraping bölümü:
1. URL'leri girin (her satıra bir tane)
2. "Web Sitelerini Tara" butonuna tıklayın
3. Progress bar ile ilerlemeyi takip edin
4. Başarılı/başarısız durumları görün
```

### 3. AI Sohbet
- Scrape edilen içerik hakkında sorular sorun
- Real-time streaming ile yanıtları izleyin
- Kaynak atıflarını kontrol edin
- Chat geçmişi otomatik kaydedilir

### 4. Oturum Yönetimi
- **Yeni Sohbet**: Temiz bir oturum başlatın
- **Oturum Geçişi**: Önceki sohbetler arasında geçiş yapın
- **Otomatik Kaydetme**: Tüm konuşmalar JSON formatında saklanır

## 🛠️ Teknik Detaylar

### Desteklenen Modeller

| Model | Hız | Maliyet | Kalite | Context | Önerilen Kullanım |
|-------|-----|---------|--------|---------|------------------|
| GPT-4o Mini | ⚡⚡⚡ | 💰 | ⭐⭐⭐ | 128K | Günlük kullanım, test |
| GPT-3.5 Turbo | ⚡⚡⚡ | 💰💰 | ⭐⭐⭐ | 16K | Genel amaçlı |
| GPT-4 | ⚡ | 💰💰💰💰 | ⭐⭐⭐⭐⭐ | 8K | Yüksek kalite analiz |
| GPT-4 Turbo | ⚡⚡ | 💰💰💰 | ⭐⭐⭐⭐⭐ | 128K | Büyük context |
| GPT-4o | ⚡⚡ | 💰💰💰 | ⭐⭐⭐⭐⭐ | 128K | Multimodal, en gelişmiş |

### Ana Bağımlılıklar

```python
# Web Framework & UI
streamlit>=1.28.0

# LangChain Ekosistemi  
langchain>=0.1.0,<0.2.0
langchain-community>=0.0.10
langchain-openai>=0.0.5

# AI & Machine Learning
openai>=1.0.0,<2.0.0
faiss-cpu>=1.7.0
tiktoken>=0.5.0

# Web Scraping
beautifulsoup4>=4.12.0
requests>=2.31.0
lxml>=4.9.0

# Utilities
python-dotenv>=1.0.0
numpy>=1.24.0
pandas>=2.0.0
```

## Proje Yapısı

```
web-scraping-ai-chat/
├── main.py                    # Ana Streamlit uygulaması
├── requirements.txt           # Python bağımlılıkları
├── requirements-minimal.txt   # Minimal kurulum
├── .env                      # Environment değişkenleri
├── .gitignore               # Git ignore kuralları
├── README.md                # Bu dokümantasyon
└── chat_data/               # Otomatik oluşturulur
    ├── sessions.json        # Sohbet oturumları
    └── scraped_data/        # Cache edilmiş içerik
```

##  Sorun Giderme

### Yaygın Problemler

**Import Hatası:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**API Key Hatası:**
```bash
# .env dosyasını kontrol edin
cat .env  # OPENAI_API_KEY=sk-... görünmeli
```

**Dependency Conflict:**
```bash
# Temiz sanal ortam oluşturun
python -m venv fresh_env
source fresh_env/bin/activate
pip install -r requirements-minimal.txt
```

**FAISS Kurulum Hatası:**
```bash
pip uninstall faiss-cpu faiss-gpu
pip install faiss-cpu --no-cache-dir
```

**Streamlit Port Hatası:**
```bash
streamlit run main.py --server.port 8502
```

## Performans İpuçları

### Maliyet Optimizasyonu
```python
model = "gpt-4o-mini"     # En ekonomik
temperature = 0.3         # Tutarlı sonuçlar
streaming = True          # Hızlı feedback
```

### Kalite Optimizasyonu
```python
model = "gpt-4"          # En kaliteli
temperature = 0.7        # Dengeli yaratıcılık
max_tokens = 3000        # Detaylı yanıtlar
```

### Hız Optimizasyonu
```python
model = "gpt-3.5-turbo"  # En hızlı
chunk_size = 500         # Küçük parçalar
streaming = True         # Real-time
```


</div>
