"""
AutoDamageIQ - Paylaşılan Sabitler
===================================
Tüm modüller arasında ortak kullanılan çeviri ve sabit değerler.
"""

# Hasar tipi çevirileri
DAMAGE_TR = {
    "crack": "Çatlak",
    "dent": "Göçük",
    "glass_shatter": "Cam Kırığı",
    "lamp_broken": "Lamba Kırığı",
    "scratch": "Çizik",
    "tire_flat": "Patlak Lastik"
}

# Parça adı çevirileri
PARTS_TR = {
    "back_bumper": "Arka Tampon",
    "back_door": "Arka Kapı",
    "back_glass": "Arka Cam",
    "back_left_door": "Arka Sol Kapı",
    "back_left_light": "Arka Sol Far",
    "back_light": "Arka Far",
    "back_right_door": "Arka Sağ Kapı",
    "back_right_light": "Arka Sağ Far",
    "front_bumper": "Ön Tampon",
    "front_door": "Ön Kapı",
    "front_glass": "Ön Cam",
    "front_left_door": "Ön Sol Kapı",
    "front_left_light": "Ön Sol Far",
    "front_light": "Ön Far",
    "front_right_door": "Ön Sağ Kapı",
    "front_right_light": "Ön Sağ Far",
    "hood": "Kaput",
    "left_mirror": "Sol Ayna",
    "object": "Nesne",
    "right_mirror": "Sağ Ayna",
    "tailgate": "Bagaj Kapağı",
    "trunk": "Bagaj",
    "wheel": "Tekerlek"
}

# Geçerli parça isimleri (VLM doğrulama için)
VALID_PARTS = list(PARTS_TR.keys())

# Şiddet eşleştirmesi
SEVERITY_MAP = {
    "crack": 3,
    "dent": 3,
    "glass_shatter": 5,
    "lamp_broken": 4,
    "scratch": 2,
    "tire_flat": 4
}
