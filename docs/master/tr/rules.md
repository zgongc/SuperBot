#!/usr/bin/env python3

"""
docs/master/rules.md

SuperBot - Geliştirme Kuralları ve Standartları
Yazar: SuperBot Team
Tarih: 2025-11-12
Versiyon: 1.0.0

Bu doküman SuperBot projesinde çalışırken uyulması gereken kuralları, kodlama
standartlarını ve süreç beklentilerini tanımlar. Amaç; modüller arası
tutarlılığı, bakım kolaylığını ve kaliteyi garanti altına almaktır.
"""

# 1. Genel Prensipler

- **Plan-Önce**: Yeni geliştirmeye başlamadan önce `docs/plans/` altında ilgili
  sprint veya mimari plan güncellenmelidir.
- **Backtest-Önce**: Canlı ortama alınacak her strateji, Backtest modülünde
  başarı kriterlerini geçmiş olmalıdır.
- **Modülerlik**: `core/` servisleri ile `components/` bileşenleri paylaşılan
  kaynaklardır; modüller gevşek bağlı olacak şekilde tasarlanmalıdır.
- **Observability**: Logging ve metrikler ilk günden düşünülmeli, minimal
  seviyede bile olsa devreye alınmalıdır.

# 2. Dosya Yapısı Standartları

- Her Python modülü zorunlu olarak başlık (header) ve test bölümü (footer)
  içermelidir.
- **Header şablonu**:

```
#!/usr/bin/env python3

"""
path/to/file.py

SuperBot - Module Name
Yazar: SuperBot Team
Tarih: YYYY-MM-DD
Versiyon: X.Y.Z

Modül açıklaması (kısa ve öz)

Özellikler:
- Özellik 1
- Özellik 2
- Özellik 3

Kullanım:
    from module import Class
    instance = Class()
    result = instance.method()

Bağımlılıklar:
    - python>=3.10
    - package1>=1.0.0
    - package2 (opsiyonel)
"""
```

- **Footer şablonu**:

```
# ============================================================================
# TEST
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 ModuleName Test")
    print("=" * 60)
    # Test 1: Basic functionality
    print("Temel fonksiyon testi:")
    # Test code here
    print("   ✅ Test başarılı")
    # Test 2: Another test
    print("İkinci test:")
    # Test code here
    print("   ✅ Test başarılı")
    print("\n✅ Tüm testler tamamlandı!")
    print("=" * 60)
```

- Test bölümü sadece CLI/daemon benzeri betikler için opsiyoneldir; kütüphane
  dosyalarında zorunludur.

# 3. Kodlama Standartları

- **Dil**: Python 3.12. Tüm kod `black` formatına uyumlu tutulmalıdır.
- **Tipler**: `from __future__ import annotations` kullan; type hint'leri eksiksiz
  yaz. Pyright/ruff uyumluluğu hedeflenir.
- **İsimlendirme**:
  - Dosyalar: `snake_case.py`
  - Sınıflar: `CapWords`
  - Değişken/Fonksiyon: `snake_case`
  - Sabitler: `UPPER_SNAKE_CASE`
- **Docstring**: Her modül, sınıf ve kompleks fonksiyon için Google tarzı
  docstring yaz. Modül başına kısa özet ekle.
- **TODO**: Gerekirse `# TODO(username): açıklama` formatını kullan; ilgili
  backlog öğesine referans ver.
- **Sınıf Yapısı**: Docstring içinde sınıfın yaptığı işler ve attribute'lar
  listelenmeli. `__init__`, `initialize`, `process` örnekleri şu şablonu
  takip eder:

```
class MyManager:
    """
    Manager açıklaması

    Bu manager şu işleri yapar:
    - İş 1
    - İş 2

    Attributes:
        config: Config engine instance
        logger: Logger instance
    """

    def __init__(self, config, logger):
        """Manager'ı başlat"""
        self.config = config
        self.logger = logger
        self._initialized = False

    def initialize(self):
        """Manager'ı başlat"""
        self.logger.info("🚀 Manager başlatılıyor...")
        # Initialization code
        self._initialized = True
        self.logger.info("✅ Manager başlatıldı")

    def process(self, data):
        """
        Veriyi işle

        Args:
            data: İşlenecek veri

        Returns:
            dict: İşlenmiş sonuç
        """
        if not self._initialized:
            raise RuntimeError("Manager başlatılmamış")

        # Processing code
        return result
```

- **Hata Yönetimi**: Türkçe mesaj + emoji ile context verilmelidir;
  örnek şablon:

```
try:
    result = risky_operation()
except ConnectionError as e:
    self.logger.error(f"❌ Bağlantı hatası: {e}")
    raise
except ValueError as e:
    self.logger.warning(f"⚠️  Geçersiz değer: {e}")
    return None
except Exception as e:
    self.logger.critical(f"🚨 Beklenmeyen hata: {e}")
    raise
```

# 4. Dosya ve Dizin Kuralları

- Her yeni bileşen, plan dokümanında belirtilen dizin yapısına sadık kalmalıdır.
- `components/` altındaki dosyalar core servislerini import ederken sadece
  gerekli fonksiyonları çekmelidir.
- Geçici script veya notebook'lar `sandbox/` adlı lokal dizinde tutulmalı,
  depoya girmemelidir.
- Konfigürasyon değişiklikleri `config/main.yaml` ve
  `config/infrastructure.yaml` üzerinden yapılmalı; varsayılan değerler
  kod içerisine gömülmemelidir.
- `config` erişimi gereken her dosya `core/config_engine.py` üzerinden config
  yüklemelidir; doğrudan YAML okumak yasaktır.
- Logging veya `print` kullanacak dosyalar mutlaka `core/logger_engine.py`
  aracılığıyla logger oluşturmalı; standardizasyon dışına çıkılmamalıdır.

# 5. Logging, Emoji ve Dil Standartları

- `core/logger_engine.py` ile sağlanan logger kullanılmalıdır; `print`
  yasaktır (yalnızca CLI/daemon entry point test çıktıları hariç).
- Log seviyeleri:
  - `debug`: Geliştirici odaklı ayrıntı
  - `info`: İş akışı adımları
  - `warning`: Beklenmeyen ama tolere edilen durum
  - `error`: Toparlanabilir hata
  - `critical`: Sistem kararlılığını tehdit eden hata
- Her log, mümkünse `strategy`, `symbol`, `timeframe`, `request_id` gibi
  bağlam etiketleri içermelidir.
- Metrikler Prometheus uyumlu tutulmalı; yeni metrik eklerken `docs/plans/`
  notlarına ek yap.
- **Log Mesajları**: %100 Türkçe, emoji ile seviyeyi belirt; hatalı örnekler
  kabul edilmez.
- **Emoji Koruma**: Kodda bulunan hiçbir emoji silinmez veya değiştirilmez.
  Konsolun emoji göstermemesi kozmetik bir durumdur; çözüm için Windows'ta
  `PYTHONIOENCODING` ve `PYTHONLEGACYWINDOWSSTDIO` ortam değişkenleri `utf-8`
  olarak ayarlanabilir.

```
# ✅ Doğru
logger.debug(f"🔍 Debug verisi: {variable}")
logger.info(f"📊 İstatistik güncellendi: {count} kayıt")
logger.warning(f"⚠️  Limit aşıldı: {warning_detail}")
logger.error(f"❌ Risk sınırı ihlali: {error_message}")
logger.critical(f"🚨 Sistem hatası: {critical_issue}")

# ❌ Yanlış
logger.debug("Debug data")
logger.info("Stats updated")
logger.warning("Warning")
```

- **Emoji Rehberi**:
  - `🔍` debug/arama
  - `✅` başarı
  - `📊` istatistik
  - `🚀` başlatma
  - `⚠️` uyarı
  - `❌` hata
  - `🚨` kritik hata
  - `🛑` durdurma
  - `🔄` yeniden başlatma
  - `💾` veri kaydı
  - `📝` log kaydı
  - `🌐` network
  - `🔐` güvenlik
  - `⏱️` zamanlama
  - `💰` sermaye
  - `📂` dosya
  - `🎯` hedef
- **Yorumlar ve Exception Mesajları**: %100 Türkçe yazılır.

```
# ✅ Doğru
# Engine'i başlat ve sağlık kontrolü yap
raise ValueError("Geçersiz config parametresi")

# ❌ Yanlış
# Start the engine and perform health check
raise ValueError("Invalid config parameter")
```

# 6. Test Politikası

- Tüm yeni kodlar için pytest tabanlı test zorunludur. Dosyaya eşlik eden test
  yoksa kod PR'da bekletilir.
- Backtest senaryoları regression test olarak çalıştırılmalı; başarısız testler
  çözülmeden merge edilmez.
- Mock yerine mümkün olduğunda fixture tabanlı gerçekçi veri kullanılmalıdır.
- Testler deterministik olmalı; random bileşenlerde seed sabitlenmelidir.

# 7. Güvenlik ve Konfigürasyon

- Gizli anahtarlar `.env` veya secret manager üzerinden yönetilir; depoya
  kesinlikle plaintext olarak konmaz.
- `security_engine` master key’ini güncellemeden önce rollback planı yaz.
- Konfigürasyon değişiklikleri `config_engine` aracılığıyla yapılmalı; manuel
  yazılan config dosyaları schema doğrulamasından geçirilmelidir.

# 8. Bağımlılık Yönetimi

- Yeni bağımlılıklar eklenmeden önce tartışma aç; lisans ve uyumluluk kontrolü
  yap.
- `requirements.txt` güncellenirken tam sürüm numarası pinlenir.
- Sistem servisi gerektiren bağımlılıklar için `docs/guides/` altında kurulum
  rehberi eklenir.

# 9. Geliştirme Süreçleri

- **Branching**: `main` korumalıdır. Özellik geliştirme için
  `feature/<module>/<özellik>` formatında branch aç.
- **Commit Mesajı**: `type(scope): açıklama` (ör. `feat(trading): add live monitor`).
  `type` seti: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `build`.
- **Code Review**: PR açmadan önce testleri çalıştır; reviewer için özet ve
  kontrol listesi ekle.
- **CI/CD**: Pipeline başarısızsa öncelik onu düzeltmektir; pipeline kırık
  halde bırakılmaz.
- **Python Ortamı**: Tüm geliştirme ve test süreçlerinde `conda activate superbot`
  komutu ile `superbot` ortamı kullanılmalıdır. Alternatif ortamlar PR açmadan
  önce yeniden bu ortama geçmelidir.

# 10. AI ve Otomasyon Kullanımı

- AI destekli araçlar (ör. strateji optimizasyonu) sonuç üretmeden önce
  `docs/plans/` altındaki planlarda tanımlanmalı.
- AI çıktıları mutlaka manuel doğrulamadan geçer; otomatik üretilen kodun
  kaynak ve gerekçesi PR açıklamasına eklenir.
- FastAPI tabanlı AI servisleri için versiyonlama ve model kayıt politikası
  `docs/plans/superbot-architecture.md` ile uyumlu tutulur.

# 11. İhlal ve Revizyon

- Bu kurallara uyumsuzluk tespit edilirse ilgili geliştirici uyarılır; tekrar
  eden durumlarda kod review süreci sıkılaştırılır.
- Doküman güncelliğini korumak için her sprint sonunda gözden geçirilir;
  revizyon gerekirse versiyon numarası artırılır.

----

Bu kurallar, SuperBot projesinin sürdürülebilir ve ölçeklenebilir şekilde
gelişmesini sağlamak için tasarlanmıştır. Ekipten her üye bu rehbere uygun
çalışmakla sorumludur.

