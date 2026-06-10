import os
import io
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import sqlite3
from datetime import datetime
from PIL import Image
from ultralytics import YOLO
import docx
import re

app = FastAPI(title="Plant Disease Detection API")

THRESHOLD = 0.70
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

try:
    my_model = YOLO(MODEL_PATH)
    print("YOLO Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    my_model = None

def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    phone TEXT PRIMARY KEY,
                    first_name TEXT,
                    last_name TEXT,
                    city TEXT,
                    district TEXT,
                    neighborhood TEXT,
                    village TEXT
                 )''')
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    phone TEXT,
                    label TEXT,
                    city TEXT,
                    district TEXT,
                    timestamp TEXT
                 )''')
    c.execute("CREATE INDEX IF NOT EXISTS idx_predictions_city_district ON predictions(city, district)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_predictions_label ON predictions(label)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_users_city_district ON users(city, district)")
    try:
        c.execute("ALTER TABLE users ADD COLUMN crops TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()

init_db()

class User(BaseModel):
    phone: str
    password: str = ""
    first_name: str
    last_name: str
    city: str
    district: str
    neighborhood: str
    village: str
    crops: str = ""

class LoginRequest(BaseModel):
    phone: str
    password: str

class UpdateProfileRequest(BaseModel):
    old_phone: str
    phone: str
    first_name: str
    last_name: str
    city: str
    district: str
    neighborhood: str
    village: str
    crops: str = ""

class StatRequest(BaseModel):
    phone: str
    label: str

WORD_DOSYA_YOLU = os.path.join(BASE_DIR, "bitki tedavileri.docx")

# Word dosyasının son okunma zamanını tutuyoruz (hot-reload için)
_son_degisiklik_zamani = 0.0

def word_dosyasindan_oku(dosya_yolu):
    """Word dosyasındaki hastalık tedavilerini okur ve anahtar:tedavi sözlüğü döndürür.
    
    Beklenen başlık formatı: Türkçe Ad (İngilizce Ad) :
    Örnek: Fasulye Pası (Bean Rust) :
    Başlık tespiti: satırda parantez içinde İngilizce isim varsa başlık kabul edilir.
    Sonundaki iki nokta üst üste olsun ya da olmasın çalışır.
    """
    tedavi_sozlugu = {}
    try:
        doc = docx.Document(dosya_yolu)
        guncel_hastalik_id = None
        guncel_tedavi = []

        # Parantez içindeki metni yakalayan regex
        baslik_deseni = re.compile(r'\(([^)]+)\)')

        def baslik_mi(text):
            """Bu satır bir hastalık başlığı mı?"""
            if len(text) > 150: return None
            if ":" not in text: return None
            eslesme = baslik_deseni.search(text)
            if not eslesme: return None
            parantez_ici = eslesme.group(1).strip()
            kelimeler = parantez_ici.split()
            if len(kelimeler) >= 6 or len(kelimeler) == 0: return None
            return parantez_ici

        for p in doc.paragraphs:
            metin = p.text.strip()
            if not metin:
                continue

            en_kisim = baslik_mi(metin)
            if en_kisim:
                # Önceki hastalığı kaydet
                if guncel_hastalik_id and guncel_tedavi:
                    tedavi_sozlugu[guncel_hastalik_id] = "\n".join(guncel_tedavi).strip()

                # Parantez içindeki İngilizce adı anahtar yap
                guncel_hastalik_id = en_kisim.lower().replace(" ", "_")
                guncel_tedavi = []

                # Başlık satırında iki nokta sonrası metin varsa onu da ekle
                if ":" in metin:
                    kalani = metin.split(":", 1)[-1].strip()
                    # Parantez kısmını temizle
                    kalani = re.sub(r'\([^)]*\)', '', kalani).strip()
                    if kalani:
                        guncel_tedavi.append(kalani)
            else:
                if guncel_hastalik_id:
                    guncel_tedavi.append(metin)

        # Döngü bitince son kalan hastalığı kaydet
        if guncel_hastalik_id and guncel_tedavi:
            tedavi_sozlugu[guncel_hastalik_id] = "\n".join(guncel_tedavi).strip()

        print(f"✅ Word dosyasından {len(tedavi_sozlugu)} adet tedavi okundu: {list(tedavi_sozlugu.keys())}")
        return tedavi_sozlugu

    except Exception as e:
        print(f"❌ Word okurken hata: {e}")
        return {}


def word_yenile_gerekiyorsa():
    """Word dosyası değiştiyse tedavi verilerini otomatik yeniden yükler."""
    global TREATMENT_DATA, _son_degisiklik_zamani
    try:
        suanki_zaman = os.path.getmtime(WORD_DOSYA_YOLU)
        if suanki_zaman != _son_degisiklik_zamani:
            print("🔄 Word dosyası değişti, tedaviler yeniden yükleniyor...")
            TREATMENT_DATA = word_dosyasindan_oku(WORD_DOSYA_YOLU)
            _son_degisiklik_zamani = suanki_zaman
    except Exception as e:
        print(f"Dosya kontrol hatası: {e}")


# İlk yükleme
TREATMENT_DATA = word_dosyasindan_oku(WORD_DOSYA_YOLU)
try:
    _son_degisiklik_zamani = os.path.getmtime(WORD_DOSYA_YOLU)
except Exception:
    pass


# Model etiketleri (YOLO çıktısı) <-> Word'den parse edilen anahtarlar arasındaki köprü
# Word'deki parantez içi İngilizce ad: lowercase + alt çizgi → word_key
# Bazı model etiketleri Word başlığıyla tam eşleşmiyor, bu tablo köprü görevi görür
LABEL_TO_WORD_KEY = {
    "bean_angular_leaf_spot":   "bean_angular_leaf_spot",
    "bean_rust":                "bean_rust",
    "corn_cercospora_leaf":     "corn_cercospora_leaf_spot",
    "corn_common_rust":         "corn_common_rust",
    "corn_northern_leaf":       "corn_northern_leaf_blight",
    "lentil_ascochyta_blight":  "lentil_ascochyta_blight",
    "lentil_powdery_mildew":    "lentil_powdery_mildew",
    "lentil_rust":              "lentil_rust",
    "pea_downy_mildew_leaf":    "pea_downy_mildew",
    "pea_leafminner_leaf":      "pea_leafminer",
    "pea_powder_mildew_leaf":   "pea_powdery_mildew",
    "potato_early_blight_leaf": "potato_early_blight",
    "potato_late_blight":       "potato_late_blight",
    "sunflower_wounded_leaf":   "sunflower_wounded_leaf",
    "withered_sunflower":       "withered_sunflower",
    "Leaf_Blight":              "wheat_leaf_blight",
    "wheat_black_rust_leaf":    "wheat_black_rust",
    "wheat_blast_leaf":         "wheat_blast",
    "wheat_brown_rust_leaf":    "wheat_brown_rust",
}


def get_treatment(label: str) -> str:
    """Model etiketine göre Word'den okunan tedavi metnini döndürür."""
    # 1. Doğrudan eşleşme
    if label in TREATMENT_DATA:
        return TREATMENT_DATA[label]
    # 2. Eşleşme tablosu üzerinden
    word_key = LABEL_TO_WORD_KEY.get(label)
    if word_key and word_key in TREATMENT_DATA:
        return TREATMENT_DATA[word_key]
    # 3. Normalize ederek deneme (küçük harf + alt çizgi)
    normalized = label.lower().replace(" ", "_")
    if normalized in TREATMENT_DATA:
        return TREATMENT_DATA[normalized]
    # 4. Kısmi eşleşme (bir taraf diğerini içeriyor mu)
    for key, value in TREATMENT_DATA.items():
        if normalized in key or key in normalized:
            return value
    print(f"⚠️ '{label}' için tedavi bulunamadı. Mevcut anahtarlar: {list(TREATMENT_DATA.keys())}")
    return "Bu hastalık için sistemde kayıtlı bir tedavi bulunamadı. Lütfen uzmana danışın."


HEALTHY_CLASSES = [
    "bean_healthy_leaf", "corn_healthy_leaf", "lentil_healthy", 
    "pea_fresh_leaf", "potato_healthy_leaf", "sunflower_fresh_leaf", 
    "wheat_healthy_leaf"
]

DISEASE_NAMES = {
    "bean_angular_leaf_spot": "Fasulye Köşeli Yaprak Lekesi (Bean Angular Leaf Spot)",
    "bean_rust": "Fasulye Pası (Bean Rust)",
    "corn_cercospora_leaf": "Mısır Serkospora Yaprak Lekesi (Corn Cercospora Leaf Spot)",
    "corn_common_rust": "Mısır Pası (Corn Common Rust)",
    "corn_northern_leaf": "Mısır Kuzey Yaprak Yanıklığı (Corn Northern Leaf Blight)",
    "lentil_ascochyta_blight": "Mercimek Antraknozu / Askokita Yanıklığı (Lentil Ascochyta Blight)",
    "lentil_powdery_mildew": "Mercimek Küllemesi (Lentil Powdery Mildew)",
    "lentil_rust": "Mercimek Pası (Lentil Rust)",
    "pea_downy_mildew_leaf": "Bezelye Mildiyösü (Pea Downy Mildew)",
    "pea_leafminner_leaf": "Bezelye Yaprak Galeri Sineği (Pea Leafminer)",
    "pea_powder_mildew_leaf": "Bezelye Küllemesi (Pea Powdery Mildew)",
    "potato_early_blight_leaf": "Patates Erken Yanıklığı (Potato Early Blight)",
    "potato_late_blight": "Patates Geç Yanıklığı / Mildiyö (Potato Late Blight)",
    "sunflower_wounded_leaf": "Ayçiçeği Yaralı Yaprak (Sunflower Wounded Leaf)",
    "withered_sunflower": "Solgun/Kurumuş Ayçiçeği (Withered Sunflower)",
    "Leaf_Blight": "Buğday Yaprak Yanıklığı (Wheat Leaf Blight)",
    "wheat_black_rust_leaf": "Buğday Kara Pası (Wheat Black Rust)",
    "wheat_blast_leaf": "Buğday Yanıklığı / Blast (Wheat Blast)",
    "wheat_brown_rust_leaf": "Buğday Kahverengi Pası (Wheat Brown Rust)",
    "bean_healthy_leaf": "Sağlıklı Fasulye Yaprağı",
    "corn_healthy_leaf": "Sağlıklı Mısır Yaprağı",
    "lentil_healthy": "Sağlıklı Mercimek",
    "pea_fresh_leaf": "Sağlıklı Bezelye Yaprağı",
    "potato_healthy_leaf": "Sağlıklı Patates Yaprağı",
    "sunflower_fresh_leaf": "Sağlıklı Ayçiçeği Yaprağı",
    "wheat_healthy_leaf": "Sağlıklı Buğday Yaprağı"
}

@app.get("/")
def read_root():
    return {"message": "Plant Disease API is running."}

@app.get("/check-phone")
def check_phone(phone: str):
    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("SELECT phone FROM users WHERE phone=?", (phone,))
        user = c.fetchone()
        conn.close()
        if user:
            return {"exists": True, "message": "Bu telefon numarası zaten kayıtlı."}
        else:
            return {"exists": False, "message": "Telefon numarası kullanılabilir."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/register")
def register_user(user: User):
    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        
        # Numara zaten var mı diye kontrol et
        c.execute("SELECT phone FROM users WHERE phone=?", (user.phone,))
        existing_user = c.fetchone()
        
        if existing_user:
            conn.close()
            return JSONResponse(status_code=400, content={"error": "Phone number is already registered."})
            
        c.execute("INSERT INTO users (phone, first_name, last_name, city, district, neighborhood, village, password, crops) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (user.phone, user.first_name, user.last_name, user.city, user.district, user.neighborhood, user.village, user.password, user.crops))
        conn.commit()
        conn.close()
        return {"success": True, "message": "User registered successfully."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/login")
def login_user(req: LoginRequest):
    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("SELECT first_name, last_name, city FROM users WHERE phone=? AND password=?", (req.phone, req.password))
        user = c.fetchone()
        conn.close()
        
        if user:
            return {
                "success": True, 
                "message": f"Hoşgeldiniz, {user[0]} {user[1]}!",
                "first_name": user[0],
                "last_name": user[1],
                "city": user[2]
            }
        else:
            return {"success": False, "error": "Sisteme kayıtlı değilsiniz veya şifreniz yanlış! Lütfen Kayıt Olun."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/get-profile")
def get_profile(phone: str):
    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("SELECT phone, first_name, last_name, city, district, neighborhood, village, crops FROM users WHERE phone=?", (phone,))
        row = c.fetchone()
        conn.close()
        if row:
            return {
                "success": True,
                "phone": row[0],
                "first_name": row[1],
                "last_name": row[2],
                "city": row[3],
                "district": row[4],
                "neighborhood": row[5],
                "village": row[6],
                "crops": row[7] if len(row) > 7 else ""
            }
        else:
            return JSONResponse(status_code=404, content={"error": "Kullanıcı bulunamadı."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/update-profile")
def update_profile(req: UpdateProfileRequest):
    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        
        if req.phone != req.old_phone:
            c.execute("SELECT phone FROM users WHERE phone=?", (req.phone,))
            if c.fetchone():
                conn.close()
                return JSONResponse(status_code=400, content={"error": "Bu telefon numarası zaten başka biri tarafından kullanılıyor."})
        
        c.execute("""UPDATE users SET 
                        phone=?, 
                        first_name=?, 
                        last_name=?, 
                        city=?, 
                        district=?, 
                        neighborhood=?, 
                        village=?,
                        crops=? 
                     WHERE phone=?""",
                  (req.phone, req.first_name, req.last_name, req.city, req.district, req.neighborhood, req.village, req.crops, req.old_phone))
        
        # Upsert: if Render DB has reset and user is missing, insert them
        if c.rowcount == 0:
            c.execute("""INSERT INTO users (phone, first_name, last_name, city, district, neighborhood, village, crops) 
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                      (req.phone, req.first_name, req.last_name, req.city, req.district, req.neighborhood, req.village, req.crops))
        
        c.execute("UPDATE predictions SET phone=? WHERE phone=?", (req.phone, req.old_phone))
        
        conn.commit()
        conn.close()
        return {"success": True, "message": "Profil bilgileri başarıyla güncellendi."}
    except Exception as e:
        if 'conn' in locals() or 'conn' in globals():
            try: conn.close()
            except: pass
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/stats")
def get_stats(city: str = None, district: str = None, label: str = None):
    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        
        # 1. Kullanıcı sayıları sorgusu
        users_query = "SELECT city, district, COUNT(*) FROM users"
        users_conditions = []
        users_params = []
        if city:
            users_conditions.append("city = ?")
            users_params.append(city)
        if district:
            users_conditions.append("district = ?")
            users_params.append(district)
        
        if users_conditions:
            users_query += " WHERE " + " AND ".join(users_conditions)
        users_query += " GROUP BY city, district"
        
        c.execute(users_query, users_params)
        user_counts = [{"city": row[0], "district": row[1], "count": row[2]} for row in c.fetchall()]
        
        # 2. Tahminler / Hastalık sayıları sorgusu
        predictions_query = "SELECT city, district, label, COUNT(*) FROM predictions"
        pred_conditions = []
        pred_params = []
        if city:
            pred_conditions.append("city = ?")
            pred_params.append(city)
        if district:
            pred_conditions.append("district = ?")
            pred_params.append(district)
        if label:
            pred_conditions.append("label = ?")
            pred_params.append(label)
            
        if pred_conditions:
            predictions_query += " WHERE " + " AND ".join(pred_conditions)
        predictions_query += " GROUP BY city, district, label"
        
        c.execute(predictions_query, pred_params)
        disease_counts = [{"city": row[0], "district": row[1], "label": row[2], "count": row[3]} for row in c.fetchall()]
        
        conn.close()
        return {"user_counts_by_region": user_counts, "disease_counts_by_region": disease_counts}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/add_stat")
def add_stat(req: StatRequest):
    try:
        city = "Bilinmiyor"
        district = "Bilinmiyor"
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("SELECT city, district FROM users WHERE phone=?", (req.phone,))
        res = c.fetchone()
        if res:
            city, district = res[0], res[1]
        
        c.execute("INSERT INTO predictions (phone, label, city, district, timestamp) VALUES (?, ?, ?, ?, ?)",
                  (req.phone, req.label, city, district, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()
        return {"success": True, "message": "Stat added successfully."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/predict")
async def predict_disease(phone: str = Form(default="Bilinmiyor"), file: UploadFile = File(...)):
    if not my_model:
        return JSONResponse(status_code=500, content={"error": "Model not loaded on server."})

    # Word dosyası değiştiyse tedavileri otomatik yenile
    word_yenile_gerekiyorsa()

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        results = my_model.predict(source=image, conf=0.0, verbose=False)[0]
        
        score = 0
        label = "Tanimlanamadi"
        
        if results.probs is not None:
            score = results.probs.top1conf.item()
            label_id = results.probs.top1
            label = results.names[label_id]
            
        if score >= THRESHOLD:
            if label in HEALTHY_CLASSES:
                 treatment = "Bu bitki saglikli gorunuyor. Herhangi bir tedaviye ihtiyac yoktur."
            else:
                 treatment = get_treatment(label)
            
            formatted_disease_name = DISEASE_NAMES.get(label, label)
            
            # Veriseti klasorune kaydetme
            try:
                dataset_folder = os.path.join(BASE_DIR, "GenelVeriseti")
                os.makedirs(dataset_folder, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_name = f"{label}_{timestamp}.jpg"
                image.save(os.path.join(dataset_folder, save_name))
            except Exception as save_err:
                print("Veriseti kaydedilemedi:", save_err)

            # Veritabanına Tahmin Sonucunu Kaydetme (Bölgesel Analiz için)
            try:
                city = "Bilinmiyor"
                district = "Bilinmiyor"
                conn = sqlite3.connect('database.db')
                c = conn.cursor()
                c.execute("SELECT city, district FROM users WHERE phone=?", (phone,))
                res = c.fetchone()
                if res:
                    city, district = res[0], res[1]
                
                c.execute("INSERT INTO predictions (phone, label, city, district, timestamp) VALUES (?, ?, ?, ?, ?)",
                          (phone, label, city, district, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                conn.commit()
                conn.close()
            except Exception as db_err:
                print("Veritabanına kaydedilemedi:", db_err)

            return {
                "success": True,
                "hastalik": formatted_disease_name,
                "guven_skoru": float(score),
                "tedavi": treatment
            }
        else:
            # Tanimlanamayanlar da kaydedilebilir
            try:
                dataset_folder = os.path.join(BASE_DIR, "GenelVeriseti", "Tanimlanamayan")
                os.makedirs(dataset_folder, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_name = f"unknown_{timestamp}.jpg"
                image.save(os.path.join(dataset_folder, save_name))
            except Exception as save_err:
                print("Veriseti kaydedilemedi:", save_err)
                
            return {
                "success": False,
                "hastalik": "Tespit Edilemedi",
                "guven_skoru": float(score),
                "tedavi": "Goruntuden herhangi bir bitki hastaligi yeterli guven skorunda (%70) tespit edilemedi."
            }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ================= YÖNETİCİ (ADMIN) İŞLEMLERİ =================

@app.get("/admin/users")
def get_all_users():
    """Sistemdeki tüm kullanıcıları listeler."""
    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("SELECT phone, first_name, last_name, city FROM users")
        users = [{"phone": row[0], "name": f"{row[1]} {row[2]}", "city": row[3]} for row in c.fetchall()]
        conn.close()
        return {"total_users": len(users), "users": users}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/admin/clear_users")
def clear_all_users():
    """Hatalı kayıtları sıfırlamak için tüm veritabanını siler."""
    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("DELETE FROM users")
        conn.commit()
        conn.close()
        return {"success": True, "message": "Tüm kullanıcılar veritabanından başarıyla silindi. Temiz bir sayfa açıldı."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/reset-db")
def reset_db():
    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("DROP TABLE IF EXISTS users")
        c.execute("DROP TABLE IF EXISTS predictions")
        conn.commit()
        conn.close()
        init_db()
        return {"success": True, "message": "Database tables users and predictions dropped and re-initialized."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
