import requests
import pandas as pd
import os

# ========================================================
# 1. PERSIAPAN FOLDER & KOORDINAT STASIUN TELUK BAYUR
# ========================================================
folder_name = "Dataset_Curah_Hujan_Kota_Padang"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

print(f"📁 Folder '{folder_name}' siap!")
print("🌍 Mengambil data dari Stasiun Meteorologi Maritim Teluk Bayur...")

# Data stasiun resmi Teluk Bayur
station_number = "96161"
station_name = "Stasiun Meteorologi Maritim Teluk Bayur"
station_lat = -0.99639
station_lon = 100.37222

# ========================================================
# 2. PROSES PENARIKAN DATA DARI API NASA POWER
# ========================================================
print(f"🚀 Memulai download otomatis dataset harian 2024-2025.\n")

# Parameter waktu (harian) dari 1 Januari 2024 sampai 31 Desember 2025
start_date = "20240101"
end_date = "20251231"

try:
    # Endpoint NASA POWER untuk data harian (Daily)
    # PRECTOTCORR = Precipitation Corrected (Curah Hujan)
    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=PRECTOTCORR"
        f"&community=AG"
        f"&longitude={station_lon}"
        f"&latitude={station_lat}"
        f"&start={start_date}"
        f"&end={end_date}"
        f"&format=JSON"
    )

    # Request ke API NASA
    response = requests.get(url)
    response.raise_for_status() # Cek jika ada error HTTP
    data = response.json()

    # Ekstrak data curah hujan harian
    curah_hujan_dict = data['properties']['parameter']['PRECTOTCORR']

    # Memproses dictionary menjadi list of dictionary (Tanggal, Nilai)
    records = []
    for time_key, value in curah_hujan_dict.items():
        # Format time_key dari "YYYYMMDD" ke "YYYY-MM-DD"
        tanggal = f"{time_key[:4]}-{time_key[4:6]}-{time_key[6:8]}"

        # -999.0 adalah kode error satelit dari NASA (data kosong), kita set jadi 0
        if value == -999.0:
            value = 0.0

        records.append({
            "Tanggal": tanggal,
            "Curah_Hujan": value
        })

    # Jadikan DataFrame
    df = pd.DataFrame(records)

    # Simpan ke CSV satu file saja
    file_path = os.path.join(folder_name, "dataset_curah_hujan_teluk_bayur_2024_2025.csv")
    df.to_csv(file_path, index=False)

    print(f"✅ Sukses: {file_path}")
    print(f"📅 Total baris data: {len(df)}")

except Exception as e:
    print(f"❌ Gagal mengambil data Teluk Bayur: {e}")

print("\n" + "="*50)
print("🎉 SELESAI! Dataset harian Teluk Bayur sudah diproses.")
print(f"Cek folder '{folder_name}' di direktori kamu.")
print("="*50)