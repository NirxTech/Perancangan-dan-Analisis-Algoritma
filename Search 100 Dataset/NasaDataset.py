import requests
import pandas as pd
import numpy as np
import os
import time

# ========================================================
# 1. PERSIAPAN FOLDER & KOORDINAT (AREA SUMATERA BARAT)
# ========================================================
folder_name = "Dataset_Curah_Hujan_Sumbar"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

print(f"📁 Folder '{folder_name}' siap!")
print("🌍 Membangun 100 titik koordinat di sekitar Sumatera Barat...")

# Membuat grid 10x10 (100 titik) untuk area Sumbar
# Latitude Sumbar: ~ -2.5 (Selatan) hingga 0.5 (Utara)
# Longitude Sumbar: ~ 98.5 (Barat/Mentawai) hingga 101.5 (Timur)
lats = np.linspace(-2.5, 0.5, 10)
lons = np.linspace(98.5, 101.5, 10)

# Menyusun daftar 100 pasang koordinat
koordinat_list = []
for lat in lats:
    for lon in lons:
        koordinat_list.append((round(lat, 4), round(lon, 4)))

# ========================================================
# 2. PROSES PENARIKAN DATA DARI API NASA POWER
# ========================================================
print(f"🚀 Memulai download otomatis 100 file dataset. Estimasi waktu: ~3 menit.\n")

# Parameter waktu (Tahun 2018 - 2025 sesuai project)
start_year = "2018"
end_year = "2025"

berhasil = 0
gagal = 0

for i, (lat, lon) in enumerate(koordinat_list, start=1):
    # Endpoint NASA POWER untuk data bulanan (Monthly)
    # PRECTOTCORR = Precipitation Corrected (Curah Hujan)
    url = (
        f"https://power.larc.nasa.gov/api/temporal/monthly/point"
        f"?parameters=PRECTOTCORR"
        f"&community=AG"
        f"&longitude={lon}"
        f"&latitude={lat}"
        f"&start={start_year}"
        f"&end={end_year}"
        f"&format=JSON"
    )
    
    try:
        # Request ke API NASA
        response = requests.get(url)
        response.raise_for_status() # Cek jika ada error HTTP
        data = response.json()
        
        # Ekstrak data curah hujan bulanan
        curah_hujan_dict = data['properties']['parameter']['PRECTOTCORR']
        
        # Memproses dictionary menjadi list of dictionary (TahunBulan, Nilai)
        records = []
        for time_key, value in curah_hujan_dict.items():
            # NASA API kadang menyelipkan rata-rata tahunan dengan kode YYYY13, kita skip itu
            if not time_key.endswith("13"): 
                # Format time_key dari "YYYYMM" ke "YYYY-MM" biar sama kayak script algoritma sebelumnya
                tahun = time_key[:4]
                bulan = time_key[4:]
                bulan_tahun = f"{tahun}-{bulan}"
                
                # -999.0 adalah kode error satelit dari NASA (data kosong), kita set jadi 0
                if value == -999.0:
                    value = 0.0
                    
                records.append({
                    "Bulan_Tahun": bulan_tahun,
                    "Curah_Hujan": value
                })
        
        # Jadikan DataFrame
        df = pd.DataFrame(records)
        
        # Simpan ke CSV
        file_path = os.path.join(folder_name, f"dataset_stasiun_{i}_lat{lat}_lon{lon}.csv")
        df.to_csv(file_path, index=False)
        
        print(f"✅ [{i}/100] Sukses: {file_path}")
        berhasil += 1
        
    except Exception as e:
        print(f"❌ [{i}/100] Gagal di koordinat ({lat}, {lon}): {e}")
        gagal += 1
        
    # Jeda 1.5 detik per request agar IP kita tidak di-banned (Rate Limit) oleh server NASA
    time.sleep(1.5)

print("\n" + "="*50)
print(f"🎉 SELESAI! Berhasil mendownload: {berhasil} file, Gagal: {gagal} file.")
print(f"Cek folder '{folder_name}' di direktori kamu.")
print("="*50)