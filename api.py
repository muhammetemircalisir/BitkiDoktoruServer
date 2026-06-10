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
import urllib.request
import json

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

# Firebase Sync and Write Helpers
def sync_users_from_firebase():
    try:
        url = "https://bitkidoktoru-3b50b-default-rtdb.firebaseio.com/users.json"
        with urllib.request.urlopen(url) as response:
            data = response.read().decode("utf-8")
            if data and data != "null":
                users = json.loads(data)
                conn = sqlite3.connect('database.db')
                c = conn.cursor()
                c.execute("DELETE FROM users")
                for phone, u in users.items():
                    if not u:
                        continue
                    c.execute("""INSERT OR REPLACE INTO users (phone, first_name, last_name, city, district, neighborhood, village, password, crops) 
                                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                              (phone, u.get("first_name", ""), u.get("last_name", ""), u.get("city", ""), u.get("district", ""), u.get("neighborhood", ""), u.get("village", ""), u.get("password", ""), u.get("crops", "")))
                conn.commit()
                conn.close()
                print(f"✅ Synced {len(users)} users from Firebase to local SQLite.")
    except Exception as e:
        print(f"Error syncing users from Firebase: {e}")

def sync_predictions_from_firebase():
    try:
        url = "https://bitkidoktoru-3b50b-default-rtdb.firebaseio.com/predictions.json"
        with urllib.request.urlopen(url) as response:
            data = response.read().decode("utf-8")
            if data and data != "null":
                preds = json.loads(data)
                conn = sqlite3.connect('database.db')
                c = conn.cursor()
                c.execute("DELETE FROM predictions")
                for pred_id, p in preds.items():
                    if not p:
                        continue
                    c.execute("""INSERT OR REPLACE INTO predictions (id, phone, label, city, district, timestamp) 
                                 VALUES (?, ?, ?, ?, ?, ?)""",
                              (pred_id, p.get("phone", ""), p.get("label", ""), p.get("city", ""), p.get("district", ""), p.get("timestamp", "")))
                conn.commit()
                conn.close()
                print(f"✅ Synced {len(preds)} predictions from Firebase to local SQLite.")
    except Exception as e:
        print(f"Error syncing predictions from Firebase: {e}")

def firebase_put_user(phone: str, user_data: dict):
    try:
        url = f"https://bitkidoktoru-3b50b-default-rtdb.firebaseio.com/users/{phone}.json"
        req = urllib.request.Request(url, data=json.dumps(user_data).encode("utf-8"), method="PUT")
        urllib.request.urlopen(req)
    except Exception as e:
        print(f"Error writing user to Firebase: {e}")

def firebase_delete_user(phone: str):
    try:
        url = f"https://bitkidoktoru-3b50b-default-rtdb.firebaseio.com/users/{phone}.json"
        req = urllib.request.Request(url, method="DELETE")
        urllib.request.urlopen(req)
    except Exception as e:
        print(f"Error deleting user from Firebase: {e}")

def firebase_post_prediction(pred_data: dict) -> str:
    try:
        url = "https://bitkidoktoru-3b50b-default-rtdb.firebaseio.com/predictions.json"
        req = urllib.request.Request(url, data=json.dumps(pred_data).encode("utf-8"), method="POST")
        with urllib.request.urlopen(req) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            return res.get("name", "")
    except Exception as e:
        print(f"Error writing prediction to Firebase: {e}")
        return ""

def update_predictions_phone_in_firebase(old_phone: str, new_phone: str):
    try:
        url = "https://bitkidoktoru-3b50b-default-rtdb.firebaseio.com/predictions.json"
        with urllib.request.urlopen(url) as response:
            data = response.read().decode("utf-8")
            if data and data != "null":
                preds = json.loads(data)
                for pred_id, p in preds.items():
                    if p and p.get("phone") == old_phone:
                        url_patch = f"https://bitkidoktoru-3b50b-default-rtdb.firebaseio.com/predictions/{pred_id}.json"
                        req = urllib.request.Request(url_patch, data=json.dumps({"phone": new_phone}).encode("utf-8"), method="PATCH")
                        urllib.request.urlopen(req)
    except Exception as e:
        print(f"Error updating predictions phone in Firebase: {e}")

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
                    village TEXT,
                    password TEXT,
                    crops TEXT DEFAULT ''
                 )''')
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY,
                    phone TEXT,
                    label TEXT,
                    city TEXT,
                    district TEXT,
                    timestamp TEXT
                 )''')
    c.execute("CREATE INDEX IF NOT EXISTS idx_predictions_city_district ON predictions(city, district)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_predictions_label ON predictions(label)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_users_city_district ON users(city, district)")
    conn.commit()
    conn.close()
    
    # Sync database from Firebase Realtime Database
    sync_users_from_firebase()
    sync_predictions_from_firebase()

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
    tedavi_sozlugu = {}
    try:
        doc = docx.Document(dosya_yolu)
        guncel_hastalik_id = None
        guncel_tedavi = []
        baslik_deseni = re.compile(r'\(([^)]+)\)')

        def baslik_mi(text):
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
                if guncel_hastalik_id and guncel_tedavi:
                    tedavi_sozlugu[guncel_hastalik_id] = "\n".join(guncel_tedavi).strip()

                guncel_hastalik_id = en_kisim.lower().replace(" ", "_")
                guncel_tedavi = []

                if ":" in metin:
                    kalani = metin.split(":", 1)[-1].strip()
                    kalani = re.sub(r'\([^)]*\)', '', kalani).strip()
                    if kalani:
                        guncel_tedavi.append(kalani)
            else:
                if guncel_hastalik_id:
                    guncel_tedavi.append(metin)

        if guncel_hastalik_id and guncel_tedavi:
            tedavi_sozlugu[guncel_hastalik_id] = "\n".join(guncel_tedavi).strip()

        print(f"✅ Word dosyasından {len(tedavi_sozlugu)} adet tedavi okundu: {list(tedavi_sozlugu.keys())}")
        return tedavi_sozlugu

    except Exception as e:
        print(f"❌ Word okurken hata: {e}")
        return {}


def word_yenile_gerekiyorsa():
    global TREATMENT_DATA, _son_degisiklik_zamani
    try:
        suanki_zaman = os.path.getmtime(WORD_DOSYA_YOLU)
        if suanki_zaman != _son_degisiklik_zamani:
            print("🔄 Word dosyası değişti, tedaviler yeniden yükleniyor...")
            TREATMENT_DATA = word_dosyasindan_oku(WORD_DOSYA_YOLU)
            _son_degisiklik_zamani = suanki_zaman
    except Exception as e:
        print(f"Dosya kontrol hatası: {e}")


TREATMENT_DATA = word_dosyasindan_oku(WORD_DOSYA_YOLU)
try:
    _son_degisiklik_zamani = os.path.getmtime(WORD_DOSYA_YOLU)
except Exception:
    pass


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
    if label in TREATMENT_DATA:
        return TREATMENT_DATA[label]
    word_key = LABEL_TO_WORD_KEY.get(label)
    if word_key and word_key in TREATMENT_DATA:
        return TREATMENT_DATA[word_key]
    normalized = label.lower().replace(" ", "_")
    if normalized in TREATMENT_DATA:
        return TREATMENT_DATA[normalized]
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
        url = f"https://bitkidoktoru-3b50b-default-rtdb.firebaseio.com/users/{phone}.json"
        with urllib.request.urlopen(url) as response:
            data = response.read().decode("utf-8")
            if data and data != "null":
                return {"exists": True, "message": "Bu telefon numarası zaten kayıtlı."}
            else:
                return {"exists": False, "message": "Telefon numarası kullanılabilir."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/register")
def register_user(user: User):
    try:
        url = f"https://bitkidoktoru-3b50b-default-rtdb.firebaseio.com/users/{user.phone}.json"
        with urllib.request.urlopen(url) as response:
            data = response.read().decode("utf-8")
            if data and data != "null":
                return JSONResponse(status_code=400, content={"error": "already registered"})
            
        user_dict = {
            "phone": user.phone,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "city": user.city,
            "district": user.district,
            "neighborhood": user.neighborhood,
            "village": user.village,
            "password": user.password,
            "crops": user.crops
        }
        
        firebase_put_user(user.phone, user_dict)
        
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("""INSERT OR REPLACE INTO users (phone, first_name, last_name, city, district, neighborhood, village, password, crops) 
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                  (user.phone, user.first_name, user.last_name, user.city, user.district, user.neighborhood, user.village, user.password, user.crops))
        conn.commit()
        conn.close()
        
        return {"success": True, "message": "User registered successfully."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/login")
def login_user(req: LoginRequest):
    try:
        url = f"https://bitkidoktoru-3b50b-default-rtdb.firebaseio.com/users/{req.phone}.json"
        with urllib.request.urlopen(url) as response:
            data = response.read().decode("utf-8")
            if not data or data == "null":
                return {"success": False, "error": "Sisteme kayıtlı değilsiniz veya şifreniz yanlış! Lütfen Kayıt Olun."}
            
            user = json.loads(data)
            if user.get("password") == req.password:
                conn = sqlite3.connect('database.db')
                c = conn.cursor()
                c.execute("""INSERT OR REPLACE INTO users (phone, first_name, last_name, city, district, neighborhood, village, password, crops) 
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                          (user.get("phone"), user.get("first_name"), user.get("last_name"), user.get("city"), user.get("district"), user.get("neighborhood"), user.get("village"), user.get("password"), user.get("crops", "")))
                conn.commit()
                conn.close()
                
                return {
                    "success": True, 
                    "message": f"Hoşgeldiniz, {user.get('first_name')} {user.get('last_name')}!",
                    "first_name": user.get("first_name"),
                    "last_name": user.get("last_name"),
                    "city": user.get("city")
                }
            else:
                return {"success": False, "error": "Sisteme kayıtlı değilsiniz veya şifreniz yanlış! Lütfen Kayıt Olun."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/get-profile")
def get_profile(phone: str):
    try:
        url = f"https://bitkidoktoru-3b50b-default-rtdb.firebaseio.com/users/{phone}.json"
        with urllib.request.urlopen(url) as response:
            data = response.read().decode("utf-8")
            if not data or data == "null":
                return JSONResponse(status_code=404, content={"error": "Kullanıcı bulunamadı."})
            
            user = json.loads(data)
            return {
                "success": True,
                "phone": user.get("phone"),
                "first_name": user.get("first_name"),
                "last_name": user.get("last_name"),
                "city": user.get("city"),
                "district": user.get("district"),
                "neighborhood": user.get("neighborhood"),
                "village": user.get("village"),
                "crops": user.get("crops", "")
            }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/update-profile")
def update_profile(req: UpdateProfileRequest):
    try:
        url = f"https://bitkidoktoru-3b50b-default-rtdb.firebaseio.com/users/{req.old_phone}.json"
        password = ""
        with urllib.request.urlopen(url) as response:
            data = response.read().decode("utf-8")
            if data and data != "null":
                password = json.loads(data).get("password", "")
        
        if req.phone != req.old_phone:
            url_new = f"https://bitkidoktoru-3b50b-default-rtdb.firebaseio.com/users/{req.phone}.json"
            with urllib.request.urlopen(url_new) as response:
                data = response.read().decode("utf-8")
                if data and data != "null":
                    return JSONResponse(status_code=400, content={"error": "Bu telefon numarası zaten başka biri tarafından kullanılıyor."})
            
            firebase_delete_user(req.old_phone)
            conn = sqlite3.connect('database.db')
            c = conn.cursor()
            c.execute("DELETE FROM users WHERE phone=?", (req.old_phone,))
            conn.commit()
            conn.close()
            
        user_dict = {
            "phone": req.phone,
            "first_name": req.first_name,
            "last_name": req.last_name,
            "city": req.city,
            "district": req.district,
            "neighborhood": req.neighborhood,
            "village": req.village,
            "password": password,
            "crops": req.crops
        }
        
        firebase_put_user(req.phone, user_dict)
        
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("""INSERT OR REPLACE INTO users (phone, first_name, last_name, city, district, neighborhood, village, password, crops) 
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                  (req.phone, req.first_name, req.last_name, req.city, req.district, req.neighborhood, req.village, password, req.crops))
        
        c.execute("UPDATE predictions SET phone=? WHERE phone=?", (req.phone, req.old_phone))
        conn.commit()
        conn.close()
        
        if req.phone != req.old_phone:
            update_predictions_phone_in_firebase(req.old_phone, req.phone)
            
        return {"success": True, "message": "Profil bilgileri başarıyla güncellendi."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/stats")
def get_stats(city: str = None, district: str = None, label: str = None):
    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        
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
        
        url = f"https://bitkidoktoru-3b50b-default-rtdb.firebaseio.com/users/{req.phone}.json"
        with urllib.request.urlopen(url) as response:
            data = response.read().decode("utf-8")
            if data and data != "null":
                u = json.loads(data)
                city, district = u.get("city", "Bilinmiyor"), u.get("district", "Bilinmiyor")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pred_dict = {
            "phone": req.phone,
            "label": req.label,
            "city": city,
            "district": district,
            "timestamp": timestamp
        }
        
        firebase_id = firebase_post_prediction(pred_dict)
        
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO predictions (id, phone, label, city, district, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                  (firebase_id if firebase_id else str(datetime.now().timestamp()), req.phone, req.label, city, district, timestamp))
        conn.commit()
        conn.close()
        
        return {"success": True, "message": "Stat added successfully."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/predict")
async def predict_disease(phone: str = Form(default="Bilinmiyor"), file: UploadFile = File(...)):
    if not my_model:
        return JSONResponse(status_code=500, content={"error": "Model not loaded on server."})

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
            
            try:
                dataset_folder = os.path.join(BASE_DIR, "GenelVeriseti")
                os.makedirs(dataset_folder, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_name = f"{label}_{timestamp}.jpg"
                image.save(os.path.join(dataset_folder, save_name))
            except Exception as save_err:
                print("Veriseti kaydedilemedi:", save_err)

            try:
                city = "Bilinmiyor"
                district = "Bilinmiyor"
                
                url = f"https://bitkidoktoru-3b50b-default-rtdb.firebaseio.com/users/{phone}.json"
                with urllib.request.urlopen(url) as response:
                    data = response.read().decode("utf-8")
                    if data and data != "null":
                        u = json.loads(data)
                        city, district = u.get("city", "Bilinmiyor"), u.get("district", "Bilinmiyor")
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                pred_dict = {
                    "phone": phone,
                    "label": label,
                    "city": city,
                    "district": district,
                    "timestamp": timestamp
                }
                
                firebase_id = firebase_post_prediction(pred_dict)
                
                conn = sqlite3.connect('database.db')
                c = conn.cursor()
                c.execute("INSERT OR REPLACE INTO predictions (id, phone, label, city, district, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                          (firebase_id if firebase_id else str(datetime.now().timestamp()), phone, label, city, district, timestamp))
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
    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("DELETE FROM users")
        conn.commit()
        conn.close()
        
        # Clear Firebase users too
        try:
            url = "https://bitkidoktoru-3b50b-default-rtdb.firebaseio.com/users.json"
            req = urllib.request.Request(url, method="DELETE")
            urllib.request.urlopen(req)
        except Exception as fe:
            print("Failed to clear Firebase users:", fe)
            
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
