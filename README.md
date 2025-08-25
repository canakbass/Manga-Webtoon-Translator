Manga/Webtoon Tercüme Otomasyon Aracı
Bu Python projesi, manga ve webtoon görselleri gibi çizimlerdeki konuşma balonlarını otomatik olarak tespit etmek, içindeki İngilizce metni Türkçeye çevirmek ve çevrilen metni tekrar balonun içine yazmak için geliştirilmiş bir otomasyon aracıdır. Proje, özellikle uzun ve çok sayfalı webtoon serilerini tek seferde işleyebilecek şekilde tasarlanmıştır.

Özellikler
Konuşma Balonu Tespiti: Gelişmiş OpenCV ve kontur analizi yöntemleri kullanarak konuşma balonlarını doğru bir şekilde algılar.

Metin Tanıma (OCR): Pytesseract kütüphanesini kullanarak balonların içindeki İngilizce metinleri tanır.

Çeviri: Deep-translator kütüphanesi aracılığıyla tanınan metinleri Google Translate API'si üzerinden hızlı ve anahtarsız bir şekilde Türkçeye çevirir.

Akıllı Metin Yerleşimi: Çevrilen metni, Pillow kütüphanesi ile balonun içine en uygun font boyutunu ayarlayarak ve çok satırlı olarak yerleştirir. Balonun içini orijinal çizimi bozmadan yeniden doldurur.

Toplu İşleme ve Klasör Yönetimi: Belirtilen bir ana klasör içindeki tüm alt klasörleri tarar, her klasörü ayrı bir bölüm olarak işler ve çıktıları orijinal klasör yapısına göre kaydeder.

Büyük Görsel Desteği: Pillow'un (PIL) 65500 piksel yükseklik sınırını aşan uzun webtoon görsellerini otomatik olarak parçalara ayırıp işler, ardından çevrilmiş parçaları tekrar birleştirir ve orijinal dosya boyutunda çıktı verir.

Hata Yönetimi ve Loglama: İşlem sırasındaki hataları (çeviri hatası, metin tanıma hatası vb.) yönetir ve konsola bilgilendirici mesajlar yazdırır.

Gereksinimler
Projenin çalışması için aşağıdaki kütüphanelerin yüklü olması gerekir:

Pillow

opencv-python

numpy

pytesseract

deep_translator

Bu kütüphaneleri pip ile kurabilirsiniz:
pip install Pillow opencv-python numpy pytesseract deep-translator

Ayrıca, pytesseract için Tesseract OCR motorunun sisteminize yüklü olması gerekmektedir. Tesseract OCR kurulum sayfasından işletim sisteminize uygun sürümü indirip kurabilirsiniz. Kurulum tamamlandıktan sonra, manga_translator.py dosyasındaki pytesseract.pytesseract.tesseract_cmd satırındaki yolu kendi Tesseract kurulum yolunuzla değiştirmeniz gerekmektedir.

pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

Kullanım
Proje klasörünü indirin.

manga_translator.py dosyasını açın ve Tesseract kurulum yolunuzu güncelleyin.

Çevirmek istediğiniz görselleri (Webtoon'lar için her bölüm ayrı bir alt klasörde olacak şekilde) input_root değişkeninde belirtilen klasöre yerleştirin.

output_root değişkeninde belirtilen klasöre çevrilen görseller kaydedilecektir.

Terminalde aşağıdaki komutu çalıştırarak projeyi başlatın:
python manga_translator.py

Klasör Yapısı
/manga-translator-projesi
|-- manga_translator.py
|-- arial.ttf (önerilen font)
|-- /Giris (input_root)
|   |-- /Bolum1
|   |   |-- 01.jpg
|   |   |-- 02.jpg
|   |-- /Bolum2
|   |   |-- 01.png
|   |   |-- 02.png
|-- /Cikis (output_root)
|   |-- /Bolum1
|   |   |-- stitched_translated_01.jpg
|   |   |-- translated_01_001.jpg
|   |-- /Bolum2
|   |   |-- stitched_translated_01.jpg
|   |   |-- translated_01_001.jpg
