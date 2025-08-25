# manga_translator.py
# Manga/Webtoon görsellerindeki konuşma balonlarını tespit eden, içindeki İngilizce metni Türkçeye çevirip tekrar balona yazan, toplu ve klasörlü çalışan klasik Python pipeline.
# OpenCV, pytesseract, deep-translator, PIL kullanır. Stitching, splitting, debug, hata yönetimi içerir.

from PIL import Image, ImageDraw, ImageFont
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
from deep_translator import GoogleTranslator
import cv2
import numpy as np
import os

def ocr_image(image_path):
    """Görselden metin tespit eder."""
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang='eng')
    return text

def translate_text(text, target_lang='tr'):
    """Metni hedef dile çevirir (Google Translate ile, anahtarsız)."""
    try:
        result = GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception as e:
        print(f"Çeviri hatası: {e}")
        result = text
    return result

def add_text_to_image(image_path, text, output_path="translated_output.jpg"):
    """Çevrilen metni görsele ekler."""
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    text_width, text_height = draw.textsize(text, font=font)
    x = (img.width - text_width) // 2
    y = 10
    draw.rectangle([(x-10, y-10), (x+text_width+10, y+text_height+10)], fill="white")
    draw.text((x, y), text, fill="black", font=font)
    img.save(output_path)
    print(f"Çevrili metin görsele eklendi: {output_path}")

def detect_speech_balloons(image_path):
    """Görseldeki konuşma balonlarını tespit eder ve koordinatlarını döndürür. Yüzdelik kontur kutusu ile."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    balloons = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 80 and h > 40 and w/h < 3 and h/w < 3:
            xs = cnt[:, 0, 0]
            ys = cnt[:, 0, 1]
            x1 = int(np.percentile(xs, 10))
            x2 = int(np.percentile(xs, 90))
            y1 = int(np.percentile(ys, 10))
            y2 = int(np.percentile(ys, 90))
            new_w = x2 - x1
            new_h = y2 - y1
            if new_w > 30 and new_h > 20:
                balloons.append((x1, y1, new_w, new_h))
    return balloons

def detect_speech_balloons_with_contours(
    image_path,
    min_w=40, min_h=20, max_w=2000, max_h=2000,
    min_area=800, max_area=999999,
    max_ratio=4.0
):
    """Görseldeki konuşma balonlarını tespit eder, hem bounding box hem kontur döndürür. Filtreler daha gevşek."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    balloons = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        ratio = max(w/h, h/w)
        if w < min_w or h < min_h or w > max_w or h > max_h:
            continue
        if area < min_area or area > max_area:
            continue
        if ratio > max_ratio:
            continue
        xs = cnt[:, 0, 0]
        ys = cnt[:, 0, 1]
        x1 = int(np.percentile(xs, 10))
        x2 = int(np.percentile(xs, 90))
        y1 = int(np.percentile(ys, 10))
        y2 = int(np.percentile(ys, 90))
        new_w = x2 - x1
        new_h = y2 - y1
        if new_w > 15 and new_h > 10:
            balloons.append({
                'box': (x1, y1, new_w, new_h),
                'contour': cnt
            })
    return balloons

def add_texts_to_balloons(image_path, texts, output_path="translated_balloons.jpg", balloons=None):
    """Balon kutusunun içini inpainting ile doldurur, ardından metni arka plansız olarak yazar."""
    img = Image.open(image_path).convert("RGB")
    if balloons is None:
        balloons = detect_speech_balloons(image_path)
    for (box, text) in zip(balloons, texts):
        x, y, w, h = box
        img = fill_balloon_inpaint(image_path, box)
        draw = ImageDraw.Draw(img)
        text = clean_translated_text(text)
        pad = 6
        box_padded = (x+pad, y+pad, w-2*pad, h-2*pad)
        font, lines, line_heights, total_text_height, line_spacing = fit_text_to_box_multiline(draw, text, box_padded)
        ty = y + pad + ((h-2*pad) - total_text_height) // 2
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = line_heights[i]
            tx = x + pad + ((w-2*pad) - text_width) // 2
            draw.text((tx, ty), line, fill="black", font=font)
            ty += text_height + line_spacing
    img.save(output_path)
    print(f"Çevrilen metinler balonlara eklendi: {output_path}")

def add_texts_to_balloons_with_contour(image_path, texts, output_path="translated_balloons.jpg", balloons=None):
    """Balonun konturunu maske ile doldurur, metni kutunun içine ortalar."""
    img = Image.open(image_path).convert("RGB")
    if balloons is None:
        balloons = detect_speech_balloons_with_contours(image_path)
    for balloon, text in zip(balloons, texts):
        box = balloon['box']
        contour = balloon['contour']
        img = fill_balloon_contour_mask(image_path, contour)
        draw = ImageDraw.Draw(img)
        text = clean_translated_text(text)
        x, y, w, h = box
        pad = 6
        box_padded = (x+pad, y+pad, w-2*pad, h-2*pad)
        font, lines, line_heights, total_text_height, line_spacing = fit_text_to_box_multiline(draw, text, box_padded)
        ty = y + pad + ((h-2*pad) - total_text_height) // 2
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = line_heights[i]
            tx = x + pad + ((w-2*pad) - text_width) // 2
            draw.text((tx, ty), line, fill="black", font=font)
            ty += text_height + line_spacing
    img.save(output_path)
    print(f"Çevrilen metinler balonlara eklendi: {output_path}")

def draw_balloons_on_image(image_path, balloons, output_path="balloon_detected.jpg"):
    """Tespit edilen balonların etrafına kırmızı dikdörtgen çizer."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for (x, y, w, h) in balloons:
        draw.rectangle([(x, y), (x+w, y+h)], outline="red", width=3)
    img.save(output_path)
    print(f"Balon tespit görseli kaydedildi: {output_path}")

def crop_with_padding(img, x, y, w, h, padding=10):
    """Kırpılan kutunun etrafına padding ekler, sınırları aşmaz."""
    left = max(x - padding, 0)
    upper = max(y - padding, 0)
    right = min(x + w + padding, img.width)
    lower = min(y + h + padding, img.height)
    return img.crop((left, upper, right, lower))

def fill_balloon_contour(image_path, box, fill_color=(255,255,255)):
    """Belirtilen kutu içindeki balon konturunun sadece içini beyazla doldurur, kenar çizgiyi korur."""
    img_cv = cv2.imread(image_path)
    x, y, w, h = box
    roi = img_cv[y:y+h, x:x+w].copy()
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, roi_thresh = cv2.threshold(roi_gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    for cnt in contours:
        cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
    roi_bg = roi.copy()
    roi_bg[mask == 255] = fill_color
    img_cv[y:y+h, x:x+w] = roi_bg
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    return img_pil

def fill_balloon_box_with_sample_color(image_path, box):
    """Kutunun ortasındaki pikselin rengini alıp, kutunun içini o renkle doldurur."""
    img = Image.open(image_path).convert("RGB")
    x, y, w, h = box
    center_x = x + w // 2
    center_y = y + h // 2
    sample_color = img.getpixel((center_x, center_y))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(x+2, y+2), (x+w-2, y+h-2)], fill=sample_color)
    return img

def fill_balloon_inpaint(image_path, box):
    """Belirtilen kutu içindeki balonun içini inpainting ile doğal şekilde doldurur."""
    img_cv = cv2.imread(image_path)
    x, y, w, h = box
    roi = img_cv[y:y+h, x:x+w].copy()
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(roi_gray, 220, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    roi_inpaint = cv2.inpaint(roi, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    img_cv[y:y+h, x:x+w] = roi_inpaint
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    return img_pil

def fill_balloon_contour_mask(image_path, contour, fill_color=(255,255,255), outline_color=(0,0,0), outline_thickness=2):
    """Verilen konturun içini doldurur, ardından kontur çizgisini outline_color ile tekrar çizer."""
    img_cv = cv2.imread(image_path)
    cv2.drawContours(img_cv, [contour], -1, fill_color, thickness=cv2.FILLED)
    cv2.drawContours(img_cv, [contour], -1, outline_color, thickness=outline_thickness)
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    return img_pil

def clean_translated_text(text):
    """Çeviri sonrası baştaki ve sondaki gereksiz karakterleri temizler. None gelirse boş string döndürür."""
    import re
    if not text or not isinstance(text, str):
        return ''
    return re.sub(r'^[\|\-:•.\s]+|[\|\-:•.\s]+$', '', text)

def wrap_text(draw, text, font, max_width):
    """Metni, verilen genişliğe sığacak şekilde satırlara böler."""
    words = text.split()
    lines = []
    current_line = ''
    for word in words:
        test_line = current_line + (' ' if current_line else '') + word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        line_width = bbox[2] - bbox[0]
        if line_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines

def fit_text_to_box_multiline(draw, text, box, font_path="arial.ttf", max_font_size=40, min_font_size=10):
    """Font boyutunu binary search ile optimize ederek, metni balona en büyük ve sığacak şekilde yerleştirir."""
    x, y, w, h = box
    best = (min_font_size, [], [], 0, 4)
    low, high = min_font_size, max_font_size
    while low <= high:
        font_size = (low + high) // 2
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()
        lines = wrap_text(draw, text, font, w - 10)
        line_heights = [draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] for line in lines]
        line_spacing = max(font_size // 4, 4)
        total_text_height = sum(line_heights) + (len(lines)-1)*line_spacing
        max_line_width = max([draw.textbbox((0, 0), line, font=font)[2] - draw.textbbox((0, 0), line, font=font)[0] for line in lines]) if lines else 0
        if max_line_width <= w - 10 and total_text_height <= h - 10:
            best = (font_size, lines, line_heights, total_text_height, line_spacing)
            low = font_size + 1
        else:
            high = font_size - 1
    font_size, lines, line_heights, total_text_height, line_spacing = best
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
    return font, lines, line_heights, total_text_height, line_spacing

def stitch_images_vertical(image_paths, output_path):
    """Görselleri alt alta birleştirip tek bir uzun görsel oluşturur."""
    images = [Image.open(p) for p in image_paths]
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    max_width = max(widths)
    total_height = sum(heights)
    stitched_img = Image.new('RGB', (max_width, total_height), (255,255,255))
    y_offset = 0
    for img in images:
        stitched_img.paste(img, (0, y_offset))
        y_offset += img.height
    stitched_img.save(output_path)
    print(f"Birleştirilmiş görsel kaydedildi: {output_path}")
    return heights

def split_image_vertical(stitched_image_path, original_heights, output_dir, prefix="split_"):
    """Birleştirilmiş uzun görseli, orijinal yüksekliklere göre tekrar parçalara böler."""
    img = Image.open(stitched_image_path)
    y_offset = 0
    for idx, h in enumerate(original_heights):
        crop = img.crop((0, y_offset, img.width, y_offset + h))
        out_path = f"{output_dir}/{prefix}{idx+1:03d}.jpg"
        crop.save(out_path)
        print(f"Parça kaydedildi: {out_path}")
        y_offset += h

def stitch_images_vertical_limited(image_paths, output_dir, prefix="stitched", max_height=65000):
    """Görselleri alt alta birleştirip, toplam yükseklik max_height'ı aşarsa parçalara böler."""
    images = [Image.open(p) for p in image_paths]
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    max_width = max(widths)
    stitched_parts = []
    part_imgs = []
    part_heights = []
    current_height = 0
    part_idx = 1
    for img, h in zip(images, heights):
        if current_height + h > max_height and part_imgs:
            stitched_img = Image.new('RGB', (max_width, current_height), (255,255,255))
            y_offset = 0
            for im in part_imgs:
                stitched_img.paste(im, (0, y_offset))
                y_offset += im.height
            out_path = os.path.join(output_dir, f"{prefix}_{part_idx:02d}.jpg")
            stitched_img.save(out_path)
            print(f"Birleştirilmiş görsel kaydedildi: {out_path}")
            stitched_parts.append((out_path, part_heights.copy()))
            part_imgs = []
            part_heights = []
            current_height = 0
            part_idx += 1
        part_imgs.append(img)
        part_heights.append(h)
        current_height += h
    if part_imgs:
        stitched_img = Image.new('RGB', (max_width, current_height), (255,255,255))
        y_offset = 0
        for im in part_imgs:
            stitched_img.paste(im, (0, y_offset))
            y_offset += im.height
        out_path = os.path.join(output_dir, f"{prefix}_{part_idx:02d}.jpg")
        stitched_img.save(out_path)
        print(f"Birleştirilmiş görsel kaydedildi: {out_path}")
        stitched_parts.append((out_path, part_heights.copy()))
    return stitched_parts

def preprocess_for_ocr(img):
    """OCR öncesi crop'u griye çevirip kontrastı artır, ardından threshold uygula."""
    from PIL import ImageEnhance
    img = img.convert('L')
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    img = img.point(lambda x: 0 if x < 180 else 255, '1')
    return img

def process_webtoon_folders(input_root, output_root):
    """Tüm alt klasörlerdeki görselleri birleştir, çevir, tekrar böl ve çıkışa kaydet."""
    for dirpath, dirnames, filenames in os.walk(input_root):
        image_files = sorted([f for f in filenames if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if not image_files:
            continue
        abs_image_paths = [os.path.join(dirpath, f) for f in image_files]
        rel_dir = os.path.relpath(dirpath, input_root)
        out_dir = os.path.join(output_root, rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        stitched_parts = stitch_images_vertical_limited(abs_image_paths, out_dir, prefix='stitched', max_height=65000)
        for idx, (stitched_path, heights) in enumerate(stitched_parts, 1):
            balloons = detect_speech_balloons_with_contours(
                stitched_path,
                min_w=40, min_h=20, max_w=2000, max_h=2000,
                min_area=800, max_area=999999, max_ratio=4.0
            )
            print(f"{stitched_path} için {len(balloons)} balon bulundu.")
            debug_balloons_img = os.path.join(os.path.dirname(stitched_path), f'debug_balloons_{idx:02d}.jpg')
            draw_balloons_on_image(stitched_path, [b['box'] for b in balloons], output_path=debug_balloons_img)
            if len(balloons) > 50:
                print(f"UYARI: Çok fazla balon bulundu ({len(balloons)}). Filtreleri daha da sıkılaştırmak isteyebilirsiniz!")
            texts = []
            valid_balloons = []
            for i, balloon in enumerate(balloons, 1):
                box = balloon['box']
                x, y, w, h = box
                img = Image.open(stitched_path)
                crop = crop_with_padding(img, x, y, w, h, padding=12)
                crop_ocr = preprocess_for_ocr(crop)
                custom_config = r'--oem 3 --psm 6'
                text = pytesseract.image_to_string(crop_ocr, lang='eng', config=custom_config)
                text = ' '.join([line.strip() for line in text.splitlines() if line.strip()])
                if not text.strip():
                    continue
                ceviri = translate_text(text, target_lang='tr')
                if not ceviri or not isinstance(ceviri, str) or not ceviri.strip():
                    continue
                texts.append(ceviri)
                valid_balloons.append(balloon)
                if len(balloons) > 10 and i % 10 == 0:
                    print(f"  -> {i}/{len(balloons)} balon işlendi...")
            print(f"{stitched_path} için {len(texts)} balonda metin tespit edildi ve çevrildi.")
            if texts:
                out_translated = os.path.join(out_dir, f'stitched_translated_{idx:02d}.jpg')
                add_texts_to_balloons_with_contour(stitched_path, texts, output_path=out_translated, balloons=valid_balloons)
                split_image_vertical(out_translated, heights, out_dir, prefix=f"translated_{idx:02d}_")
            else:
                print(f"{stitched_path} için hiç balon/metin tespit edilemedi, sadece stitched görsel kaydedildi.")
                split_image_vertical(stitched_path, heights, out_dir, prefix=f"split_{idx:02d}_")

if __name__ == "__main__":
    input_folder = r"C:/test/translate/GorselTest/Giris"
    output_folder = r"C:/test/translate/GorselTest/Cikis"
    process_webtoon_folders(input_folder, output_folder)
