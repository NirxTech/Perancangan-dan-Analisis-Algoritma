import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
import warnings
import os

# Menonaktifkan warning agar output di terminal lebih bersih
warnings.filterwarnings('ignore')

# ==========================================
# 1. MEMBACA DATASET & PREPROCESSING (.xlsx)
# ==========================================
# Sesuaikan dengan nama file Excel kamu
# Argparse untuk input file dan horizon prediksi
parser = argparse.ArgumentParser(description='Analisis Curah Hujan: Holt-Winters, Decision Tree, Linear Regression')
parser.add_argument('file', nargs='?', default='Dataset Curah Hujan.xlsx', help='Nama file Excel dataset (default: Dataset Curah Hujan.xlsx)')
parser.add_argument('-H', '--horizon', type=int, default=12, help='Horizon prediksi dalam bulan (mis. 6, 12)')
args = parser.parse_args()
file_name = args.file
FORECAST_HORIZON = max(1, int(args.horizon))

try:
    # Helper untuk membaca Excel historis dan mengubahnya ke time-series
    bulan_map = {
        "Januari": "01", "Februari": "02", "Maret": "03", "April": "04",
        "Mei": "05", "Juni": "06", "Juli": "07", "Agustus": "08",
        "September": "09", "Oktober": "10", "November": "11", "Desember": "12"
    }

    def to_timeseries_from_matrix(raw_df):
        data_start = None
        bulan_keys = set([b.title() for b in bulan_map.keys()])

        for idx, row in raw_df.iterrows():
            try:
                val = str(row.iloc[0]).strip().title()
            except Exception:
                val = ''
            if val in bulan_keys:
                data_start = idx
                break

        if data_start is None:
            for idx, row in raw_df.iterrows():
                try:
                    val = str(row.iloc[0]).strip().lower()
                except Exception:
                    val = ''
                if val.startswith('jan'):
                    data_start = idx
                    break

        year_header = None
        if data_start is not None:
            for j in range(data_start - 1, -1, -1):
                row = raw_df.iloc[j].astype(str)
                if row.str.contains(r"\d{4}").any():
                    year_header = j
                    break

        if year_header is None and data_start is not None and data_start > 0:
            year_header = data_start - 1

        if data_start is None:
            raise ValueError('Tidak dapat menemukan baris data bulan pada file Excel.')

        if year_header is not None:
            header_row = raw_df.iloc[year_header].fillna('').astype(str)
            df_data = raw_df.iloc[data_start:].copy().reset_index(drop=True)
            df_data.columns = header_row.values
        else:
            df_data = raw_df.iloc[data_start:].copy().reset_index(drop=True)
            cols = ['Bulan'] + [f'Col{i}' for i in range(1, df_data.shape[1])]
            df_data.columns = cols

        kolom_bulan = df_data.columns[0]
        df_melted = df_data.melt(id_vars=[kolom_bulan], var_name='Tahun', value_name='Curah_Hujan')
        df_melted['Curah_Hujan'] = df_melted['Curah_Hujan'].astype(str).str.replace(',', '.', regex=False).str.strip()
        df_melted['Curah_Hujan'] = pd.to_numeric(df_melted['Curah_Hujan'], errors='coerce')
        df_melted[kolom_bulan] = df_melted[kolom_bulan].astype(str).str.strip().str.title()
        df_melted['Bulan_Angka'] = df_melted[kolom_bulan].map(bulan_map)
        df_melted = df_melted.dropna(subset=['Bulan_Angka', 'Curah_Hujan'])
        df_melted['Tahun'] = df_melted['Tahun'].astype(str).str.strip()
        df_melted['Tahun_Clean'] = df_melted['Tahun'].str.extract(r'(\d{4})')
        mask_two_digit = df_melted['Tahun_Clean'].isna()
        df_melted.loc[mask_two_digit, 'Tahun_Clean'] = df_melted.loc[mask_two_digit, 'Tahun'].str.extract(r'(\d{2})')
        df_melted.loc[mask_two_digit & df_melted['Tahun_Clean'].notna(), 'Tahun_Clean'] = (
            '20' + df_melted.loc[mask_two_digit & df_melted['Tahun_Clean'].notna(), 'Tahun_Clean']
        )
        df_melted = df_melted.dropna(subset=['Tahun_Clean'])
        df_melted['Tahun'] = df_melted['Tahun_Clean']
        df_melted = df_melted.drop(columns=['Tahun_Clean'])
        df_melted['Bulan_Tahun'] = pd.to_datetime(df_melted['Tahun'] + '-' + df_melted['Bulan_Angka'], format='%Y-%m', errors='coerce')
        df_melted = df_melted.dropna(subset=['Bulan_Tahun'])
        return df_melted[['Bulan_Tahun', 'Curah_Hujan']]

    def load_csv_timeseries(csv_path):
        csv_df = pd.read_csv(csv_path)
        if csv_df.empty:
            return None

        lower_cols = {col.lower(): col for col in csv_df.columns}
        time_col = None
        value_col = None

        for candidate in ['bulan_tahun', 'tanggal', 'date', 'waktu', 'time']:
            if candidate in lower_cols:
                time_col = lower_cols[candidate]
                break
        for candidate in ['curah_hujan', 'curah hujan', 'rainfall', 'nilai', 'value', 'mm']:
            if candidate in lower_cols:
                value_col = lower_cols[candidate]
                break

        if time_col is None:
            time_col = csv_df.columns[0]
        if value_col is None:
            numeric_cols = csv_df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                value_col = numeric_cols[0]
            else:
                candidate_cols = [c for c in csv_df.columns if c != time_col]
                value_col = candidate_cols[0] if candidate_cols else csv_df.columns[0]

        out = csv_df[[time_col, value_col]].copy()
        out.columns = ['Bulan_Tahun', 'Curah_Hujan']
        out['Bulan_Tahun'] = pd.to_datetime(out['Bulan_Tahun'], errors='coerce')
        out['Curah_Hujan'] = out['Curah_Hujan'].astype(str).str.replace(',', '.', regex=False).str.replace(r'[^0-9\.-]', '', regex=True)
        out['Curah_Hujan'] = pd.to_numeric(out['Curah_Hujan'], errors='coerce')
        out = out.dropna(subset=['Bulan_Tahun', 'Curah_Hujan'])
        return out

    # 1) Load historis dari Excel
    excel_raw = pd.read_excel(file_name, header=None)
    df_excel = to_timeseries_from_matrix(excel_raw)

    # 2) Load semua CSV dari folder Dataset_Curah_Hujan_Sumbar
    csv_folder = 'Dataset_Curah_Hujan_Sumbar'
    csv_frames = []
    if os.path.isdir(csv_folder):
        for item in os.listdir(csv_folder):
            if item.lower().endswith('.csv'):
                csv_path = os.path.join(csv_folder, item)
                try:
                    loaded_csv = load_csv_timeseries(csv_path)
                    if loaded_csv is not None and not loaded_csv.empty:
                        csv_frames.append(loaded_csv)
                except Exception as csv_error:
                    print(f"⚠️ Gagal membaca CSV '{item}': {csv_error}")
    else:
        print(f"⚠️ Folder '{csv_folder}' tidak ditemukan. Hanya data Excel yang digunakan.")

    # 3) Gabungkan Excel + semua CSV, lalu agregasi mean per waktu
    all_frames = [df_excel] + csv_frames if csv_frames else [df_excel]
    df_combined = pd.concat(all_frames, ignore_index=True)
    df_combined['Bulan_Tahun'] = pd.to_datetime(df_combined['Bulan_Tahun'], errors='coerce')
    df_combined['Curah_Hujan'] = pd.to_numeric(df_combined['Curah_Hujan'], errors='coerce')
    df_combined = df_combined.dropna(subset=['Bulan_Tahun', 'Curah_Hujan'])
    df_combined = df_combined.groupby('Bulan_Tahun', as_index=False)['Curah_Hujan'].mean().sort_values('Bulan_Tahun').reset_index(drop=True)

    # Finalisasi dataframe yang siap pakai untuk algoritma di bawah
    df = df_combined

    print("✅ Data Excel + CSV berhasil dimuat, digabung, dan diagregasi!\n")
    # Ringkasan singkat agar output lebih rapi
    try:
        print(f"Jumlah observasi setelah preprocessing: {len(df)}")
        if len(df) > 0:
            print(f"Rentang tanggal: {df['Bulan_Tahun'].min().strftime('%Y-%m')} sampai {df['Bulan_Tahun'].max().strftime('%Y-%m')}")
            print('\nContoh data (10 baris pertama):')
            print(df.head(10).to_string(index=False))
            print(f"\nSumber CSV yang dibaca: {len(csv_frames)} file")
    except Exception:
        pass

    # Jika hasil preprocessing kosong, tampilkan informasi diagnostik dan hentikan program
    if df.empty:
        print("⚠️ Hasil preprocessing menghasilkan dataframe kosong. Periksa format file Excel Anda.")
        try:
            print('\n--- Cuplikan data mentah (df_raw.head()) ---')
            print(df_raw.head().to_string(index=False))
        except Exception:
            pass
        try:
            print('\n--- Struktur kolom pada file Excel ---')
            print(list(excel_raw.columns))
        except Exception:
            pass
        print('\nPetunjuk: Pastikan Excel berisi matriks bulan-tahun dan CSV memiliki kolom waktu + curah hujan.')
        print('Jika format CSV berbeda, sesuaikan nama kolom waktu/nilai pada helper loader.')
        exit()

except FileNotFoundError:
    print(f"❌ ERROR: File '{file_name}' tidak ditemukan.")
    print("Pastikan file .xlsx berada di dalam folder yang sama dengan script ini.")
    exit()
except ImportError:
    print("❌ ERROR: Library 'openpyxl' belum terinstall.")
    print("Jalankan perintah ini di terminal: pip install openpyxl")
    exit()
except Exception as e:
    print(f"⚠️ Terjadi kesalahan saat memproses data: {e}")
    exit()

# ==========================================
# 2. ALGORITMA 1: HOLT-WINTERS (PREDIKSI)
# ==========================================
print("="*50)
print("📌 ALGORITMA 1: HOLT-WINTERS (PREDIKSI)")
print("="*50)

# Siapkan series dengan index datetime
series = df.set_index('Bulan_Tahun')['Curah_Hujan']
n = len(series)

prediksi_hw = None
hw_method_desc = ''

if n == 0:
    print("⚠️ Data kosong — tidak ada yang bisa diproses untuk prediksi.")
else:
    try:
        # Jika tersedia setidaknya 2 siklus musiman (mis. 24 bulan), gunakan komponen musiman
        if n >= 24:
            hw_model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=12).fit()
            hw_method_desc = 'Holt-Winters (musiman 12 bulan)'
        # Jika data sedikit tapi memiliki tren (>=3 titik), gunakan model dengan tren saja
        elif n >= 3:
            hw_model = ExponentialSmoothing(series, trend='add', seasonal=None).fit()
            hw_method_desc = 'Holt (tren saja, tanpa musiman)'
        else:
            raise ValueError('Data terlalu sedikit untuk pemodelan time-series (minimal 3 observasi dibutuhkan).')

        # Prediksi sesuai horizon yang diminta
        prediksi_hw = hw_model.forecast(FORECAST_HORIZON)
        print(f"Metode: {hw_method_desc}")
        print(f"Prediksi Curah Hujan {FORECAST_HORIZON} Bulan Kedepan (mm):")
        for i, val in enumerate(prediksi_hw, 1):
            val_bersih = max(0, float(val))
            print(f"Bulan ke-{i}: {val_bersih:.2f} mm")

        # Metrik in-sample (jika tersedia fittedvalues)
        try:
            fitted = hw_model.fittedvalues
            fitted = pd.Series(fitted, index=series.index)
            mae = mean_absolute_error(series, fitted)
            rmse = mean_squared_error(series, fitted, squared=False)
            print(f"\nMetrik in-sample Holt-Winters: MAE = {mae:.3f}, RMSE = {rmse:.3f}")
        except Exception:
            pass

    except Exception as e:
        print(f"⚠️ Holt-Winters gagal: {e}")
        prediksi_hw = None


# ==========================================
# 3. ALGORITMA 2: DECISION TREE (KLASIFIKASI)
# ==========================================
print("\n" + "="*50)
print("📌 ALGORITMA 2: DECISION TREE (KLASIFIKASI)")
print("="*50)

# ==========================================
# Penentuan kategori berdasarkan metode Equal Interval (sesuai laporan)
# ==========================================
def compute_equal_interval_thresholds(series, k=3):
    """Hitung threshold untuk k kategori menggunakan metode equal-interval.
    Mengembalikan tuple (low_max, mid_max) sebagai batas atas untuk kategori Rendah dan Sedang.
    """
    xmin = float(series.min())
    xmax = float(series.max())
    R = xmax - xmin
    I = R / k
    # Berdasarkan laporan: interval pertama mulai dari (xmin sampai xmin+I)
    low_max = xmin + I
    mid_max = xmin + 2 * I
    return xmin, xmax, low_max, mid_max, I


# Hitung threshold berdasarkan data yang tersedia
xmin, xmax, low_max, mid_max, interval_len = compute_equal_interval_thresholds(df['Curah_Hujan'], k=3)

def label_kategori_equal_interval(x):
    if x <= low_max:
        return 'Rendah'
    elif x <= mid_max:
        return 'Sedang'
    else:
        return 'Tinggi'

# Menerapkan label ke dataset menggunakan equal-interval sesuai laporan
df['Kategori'] = df['Curah_Hujan'].apply(label_kategori_equal_interval)

# Training Decision Tree Model (aman jika ada sedikit data)
X_dt = df[['Curah_Hujan']]
y_dt = df['Kategori']
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_trained = False
try:
    if y_dt.nunique() < 2:
        raise ValueError('Hanya ada satu kelas dalam data — tidak dapat melatih Decision Tree.')
    dt_model.fit(X_dt, y_dt)
    dt_trained = True
except Exception as e:
    print(f"⚠️ Decision Tree tidak dilatih: {e}")

print("Distribusi Kategori dalam Data Historis:")
print(df['Kategori'].value_counts().to_string())
print("\nContoh Klasifikasi 5 Data Terakhir:")
print(df[['Bulan_Tahun', 'Curah_Hujan', 'Kategori']].tail().to_string(index=False))
if dt_trained:
    try:
        train_acc = dt_model.score(X_dt, y_dt)
        cv = min(5, len(df))
        if cv >= 2:
            cv_scores = cross_val_score(dt_model, X_dt, y_dt, cv=cv)
            print(f"\nDecision Tree - Training accuracy: {train_acc:.3f}, CV({cv}) accuracy mean: {cv_scores.mean():.3f}")
        else:
            print(f"\nDecision Tree - Training accuracy: {train_acc:.3f}")
    except Exception:
        pass


# ==========================================
# 4. ALGORITMA 3: LINEAR REGRESSION (TREN)
# ==========================================
print("\n" + "="*50)
print("📌 ALGORITMA 3: LINEAR REGRESSION (ANALISIS TREN)")
print("="*50)

# Membuat Index Waktu Numerik (1, 2, 3, ...) untuk regresi
df['Time_Index'] = np.arange(1, len(df) + 1)

X_lr = df[['Time_Index']]
y_lr = df['Curah_Hujan']
lr_pred_values = None

# Training Linear Regression Model menggunakan rumus manual laporan
lr_trained = False
if len(df) >= 2:
    try:
        x_vals = df['Time_Index'].astype(float)
        y_vals = df['Curah_Hujan'].astype(float)
        n_lr = len(df)
        sum_x = float(x_vals.sum())
        sum_y = float(y_vals.sum())
        sum_x2 = float((x_vals ** 2).sum())
        sum_xy = float((x_vals * y_vals).sum())
        denominator = (n_lr * sum_x2) - (sum_x ** 2)

        if denominator == 0:
            raise ValueError('Penyebut slope bernilai 0, tidak bisa menghitung regresi linear.')

        slope = ((n_lr * sum_xy) - (sum_x * sum_y)) / denominator
        intercept = (sum_y - (slope * sum_x)) / n_lr
        lr_pred_values = intercept + (slope * x_vals)
        lr_trained = True

        print(f"Persamaan Regresi : y = {intercept:.2f} + {slope:.4f}x")
        if slope > 0:
            print(f"📈 Kesimpulan Tren : MENINGKAT (Slope bernilai positif)")
        elif slope < 0:
            print(f"📉 Kesimpulan Tren : MENURUN (Slope bernilai negatif)")
        else:
            print(f"➖ Kesimpulan Tren : STABIL / FLAT (Slope = 0)")

    except Exception as e:
        print(f"⚠️ Linear Regression gagal: {e}")
else:
    print("⚠️ Linear Regression membutuhkan setidaknya 2 observasi — dilewati.")
if lr_trained:
    try:
        y_true_lr = y_lr.astype(float)
        y_pred_lr = pd.Series(lr_pred_values, index=df.index).astype(float)
        sse = float(((y_true_lr - y_pred_lr) ** 2).sum())
        sst = float(((y_true_lr - y_true_lr.mean()) ** 2).sum())
        r2 = 1 - (sse / sst) if sst != 0 else 0.0
        rmse_lr = float(np.sqrt(mean_squared_error(y_true_lr, y_pred_lr)))
        print(f"\nLinear Regression - R^2: {r2:.3f}, RMSE: {rmse_lr:.3f}")
    except Exception:
        pass


# ==========================================
# 5. VISUALISASI HASIL PENGUJIAN
# ==========================================
print("\nMenyiapkan plot visualisasi... (Tutup jendela plot untuk mengakhiri program)")

# ====== Ekspor dan Ringkasan Hasil ======
try:
    # Simpan dataframe hasil dengan kategori
    out_df = df.copy()
    out_df.to_csv('results_summary.csv', index=False)
    print("\n✅ Hasil ringkasan disimpan di: results_summary.csv")
except Exception as e:
    print(f"⚠️ Gagal menyimpan results_summary.csv: {e}")

if prediksi_hw is not None:
    try:
        # buat future dates untuk horizon prediksi
        future_dates = pd.date_range(start=df['Bulan_Tahun'].iloc[-1] + pd.DateOffset(months=1), periods=FORECAST_HORIZON, freq='MS')
        pred_df = pd.DataFrame({'Bulan_Tahun': future_dates, 'Prediksi_Curah_Hujan': [max(0, float(v)) for v in prediksi_hw]})
        pred_df.to_csv('prediksi_hw.csv', index=False)
        print("✅ Prediksi Holt-Winters disimpan di: prediksi_hw.csv")
    except Exception as e:
        print(f"⚠️ Gagal menyimpan prediksi_hw.csv: {e}")
    # Penjelasan prediksi per-bulan dibandingkan rata-rata historis untuk bulan yang sama
    try:
        hist_monthly = series.groupby(series.index.month).mean()
        overall_mean = series.mean()
        print('\nPenjelasan prediksi per-bulan:')
        for idx, row in pred_df.iterrows():
            m = row['Bulan_Tahun'].month
            yr = row['Bulan_Tahun'].year
            pred_val = row['Prediksi_Curah_Hujan']
            hist_avg = hist_monthly.get(m, overall_mean)
            diff = pred_val - hist_avg
            direction = 'lebih tinggi' if diff > 0 else ('lebih rendah' if diff < 0 else 'sama dengan')
            print(f"- {row['Bulan_Tahun'].strftime('%Y-%m')}: prediksi {pred_val:.2f} mm; rata-rata historis bulan ini {hist_avg:.2f} mm -> {direction} sebesar {abs(diff):.2f} mm")
    except Exception:
        pass

    # Agregasi per tahun (jumlah curah hujan per tahun dari prediksi)
    try:
        pred_df['Year'] = pred_df['Bulan_Tahun'].dt.year
        pred_df['Kategori'] = pred_df['Prediksi_Curah_Hujan'].apply(label_kategori_equal_interval)
        yearly = pred_df.groupby('Year')['Prediksi_Curah_Hujan'].sum().reset_index()
        hist_yearly = series.groupby(series.index.year).sum()
        hist_yearly_mean = hist_yearly.mean() if len(hist_yearly) > 0 else None
        print('\nPrediksi agregat per-tahun (dari horizon yang diminta):')
        for _, r in yearly.iterrows():
            if hist_yearly_mean is not None:
                diff = r['Prediksi_Curah_Hujan'] - hist_yearly_mean
                direction = 'lebih tinggi' if diff > 0 else ('lebih rendah' if diff < 0 else 'sama dengan')
                print(f"- Tahun {int(r['Year'])}: total prediksi {r['Prediksi_Curah_Hujan']:.2f} mm; rata-rata tahunan historis {hist_yearly_mean:.2f} mm -> {direction} sebesar {abs(diff):.2f} mm")
            else:
                print(f"- Tahun {int(r['Year'])}: total prediksi {r['Prediksi_Curah_Hujan']:.2f} mm")
        print('\nKategori prediksi per bulan:')
        print(pred_df[['Bulan_Tahun', 'Prediksi_Curah_Hujan', 'Kategori']].to_string(index=False))
    except Exception:
        pass

    # ------- Backtesting sederhana (walk-forward) untuk Holt-Winters jika data cukup -------
    def rolling_backtest(series, h, initial_train=None):
        """Simple expanding-window backtest that forecasts h steps and compares to actuals.
        Returns dict with MAE and RMSE if ran, otherwise None.
        """
        n = len(series)
        if initial_train is None:
            initial_train = max(24, int(n * 0.7))
        if n < initial_train + h:
            return None
        preds = []
        trues = []
        i = initial_train
        # step by h to reduce computation
        while i + h <= n:
            train = series.iloc[:i]
            try:
                if len(train) >= 24:
                    model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12).fit()
                elif len(train) >= 3:
                    model = ExponentialSmoothing(train, trend='add', seasonal=None).fit()
                else:
                    break
                f = model.forecast(h)
                preds.extend(list(f))
                trues.extend(list(series.iloc[i:i+h]))
            except Exception:
                break
            i += h
        if len(preds) == 0:
            return None
        mae = mean_absolute_error(trues, preds)
        rmse = mean_squared_error(trues, preds, squared=False)
        return {'mae': mae, 'rmse': rmse, 'points': len(preds)}

    backtest_res = None
    try:
        backtest_res = rolling_backtest(series, min(FORECAST_HORIZON, 6))
        if backtest_res is not None:
            print(f"\nHasil backtest (h={min(FORECAST_HORIZON,6)}): MAE={backtest_res['mae']:.3f}, RMSE={backtest_res['rmse']:.3f} (dari {backtest_res['points']} titik)")
        else:
            print('\nBacktest tidak dilakukan: data tidak mencukupi untuk evaluasi out-of-sample sederhana.')
    except Exception:
        pass

    # ------- Visualisasi tambahan (Seaborn) -------
    try:
        sns.set_style('whitegrid')
        out_dir = 'output_plots'
        os.makedirs(out_dir, exist_ok=True)

        # 1) Boxplot per bulan (distribusi per bulan) menggunakan seaborn
        df_box = df.copy()
        df_box['Month'] = df_box['Bulan_Tahun'].dt.month
        fig_box, ax_box = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='Month', y='Curah_Hujan', data=df_box, ax=ax_box)
        ax_box.set_title('Distribusi Curah Hujan per Bulan (Boxplot)')
        ax_box.set_xlabel('Bulan (1=Jan .. 12=Des)')
        ax_box.set_ylabel('Curah Hujan (mm)')
        boxfile = os.path.join(out_dir, 'boxplot_monthly.png')
        fig_box.tight_layout()
        fig_box.savefig(boxfile)
        print(f"✅ Boxplot bulanan disimpan di: {boxfile}")

        # 2) Heatmap year vs month (pivot) menggunakan seaborn
        pivot = df.pivot_table(index=df['Bulan_Tahun'].dt.month, columns=df['Bulan_Tahun'].dt.year, values='Curah_Hujan', aggfunc='mean')
        heatfile = os.path.join(out_dir, 'heatmap_year_month.png')
        if not pivot.empty:
            fig_heat, ax_heat = plt.subplots(figsize=(12, 6))
            sns.heatmap(pivot, ax=ax_heat, cmap='viridis', cbar_kws={'label': 'Curah Hujan (mm)'}, linewidths=0.5)
            ax_heat.set_title('Heatmap Rata-rata Curah Hujan (Bulan x Tahun)')
            ax_heat.set_xlabel('Tahun')
            ax_heat.set_ylabel('Bulan')
            fig_heat.tight_layout()
            fig_heat.savefig(heatfile)
            print(f"✅ Heatmap bulanan-tahunan disimpan di: {heatfile}")
        else:
            print("⚠️ Heatmap dilewati: data pivot kosong.")

        # 3) Rata-rata bulanan (bar chart) - seaborn
        monthly_avg = df.groupby(df['Bulan_Tahun'].dt.month)['Curah_Hujan'].mean().sort_index()
        months = monthly_avg.index.tolist()
        vals = monthly_avg.values.flatten()
        fig_avg, ax_avg = plt.subplots(figsize=(10, 5))
        sns.barplot(x=months, y=vals, palette='Blues', ax=ax_avg)
        ax_avg.set_title('Rata-rata Curah Hujan per Bulan (Historis)')
        ax_avg.set_xlabel('Bulan (1=Jan .. 12=Des)')
        ax_avg.set_ylabel('Curah Hujan rata-rata (mm)')
        avgfile = os.path.join(out_dir, 'monthly_average.png')
        fig_avg.tight_layout()
        fig_avg.savefig(avgfile)
        print(f"✅ Grafik rata-rata bulanan disimpan di: {avgfile}")
        # Pastikan folder output ada
        os.makedirs('output_plots', exist_ok=True)

        # Buat figure/axes secara eksplisit untuk menghindari window kosong
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

        # --- Plot 1: Hasil Holt-Winters ---
        non_null_count = df['Curah_Hujan'].notna().sum()
        print(f"Jumlah observasi non-null untuk 'Curah_Hujan': {non_null_count}")
        print('Contoh data setelah preprocessing:')
        print(df.head(10).to_string(index=False))
        if non_null_count == 0:
            axes[0].text(0.5, 0.5, 'Tidak ada data valid untuk diplot', ha='center', va='center')
        else:
            axes[0].plot(df['Bulan_Tahun'], df['Curah_Hujan'], label='Data Aktual Historis', color='#1f77b4', marker='o', markersize=4)
            # Plot prediksi jika tersedia
            if prediksi_hw is not None:
                try:
                    future_dates = pd.date_range(start=df['Bulan_Tahun'].iloc[-1] + pd.DateOffset(months=1), periods=FORECAST_HORIZON, freq='MS')
                    prediksi_hw_bersih = [max(0, float(val)) for val in prediksi_hw]
                    axes[0].plot(future_dates, prediksi_hw_bersih, label=f'Prediksi Holt-Winters ({FORECAST_HORIZON} Bulan Kedepan)', color='#d62728', linestyle='dashed', marker='x')

                    # Tampilkan kategori prediksi langsung di grafik
                    if 'pred_df' in locals() and not pred_df.empty:
                        kategori_warna = {'Rendah': '#2ca02c', 'Sedang': '#ff7f0e', 'Tinggi': '#d62728'}
                        for _, row in pred_df.iterrows():
                            warna = kategori_warna.get(row['Kategori'], '#333333')
                            axes[0].scatter(row['Bulan_Tahun'], row['Prediksi_Curah_Hujan'], color=warna, s=50, zorder=5)
                            axes[0].annotate(
                                row['Kategori'],
                                (row['Bulan_Tahun'], row['Prediksi_Curah_Hujan']),
                                textcoords='offset points',
                                xytext=(0, 8),
                                ha='center',
                                fontsize=8,
                                color=warna
                            )
                except Exception:
                    pass

        axes[0].set_title('Prediksi Curah Hujan dengan Algoritma Holt-Winters', fontsize=14, pad=10)
        axes[0].set_xlabel('Tahun', fontsize=11)
        axes[0].set_ylabel('Curah Hujan (mm)', fontsize=11)
        axes[0].legend(loc='upper left')
        axes[0].grid(True, linestyle='--', alpha=0.6)

        # --- Plot 2: Hasil Linear Regression ---
        if non_null_count == 0:
            axes[1].text(0.5, 0.5, 'Tidak ada data valid untuk diplot', ha='center', va='center')
        else:
            axes[1].plot(df['Bulan_Tahun'], df['Curah_Hujan'], label='Data Aktual Historis', color='#7f7f7f', alpha=0.7)
            try:
                if lr_trained and lr_pred_values is not None:
                    axes[1].plot(df['Bulan_Tahun'], lr_pred_values, label=f'Garis Tren (y = {intercept:.2f} + {slope:.2f}x)', color='#2ca02c', linewidth=3)
            except Exception:
                pass

        axes[1].set_title('Analisis Tren Curah Hujan dengan Linear Regression', fontsize=14, pad=10)
        axes[1].set_xlabel('Tahun', fontsize=11)
        axes[1].set_ylabel('Curah Hujan (mm)', fontsize=11)
        axes[1].legend(loc='upper left')
        axes[1].grid(True, linestyle='--', alpha=0.6)

        # Simpan plot utama juga ke file agar bisa dicek jika window kosong
        try:
            mainfile = os.path.join('output_plots', 'main_plot.png')
            fig.tight_layout(pad=3.0)
            fig.savefig(mainfile)
            print(f"✅ Plot utama disimpan di: {mainfile}")
        except Exception as e:
            print(f"⚠️ Gagal menyimpan plot utama: {e}")

        plt.show(block=True)
    except Exception as e:
        print(f"⚠️ Error saat membuat visualisasi tambahan: {e}")

    # (Plot utama telah dibuat dan disimpan sebelumnya menggunakan objek fig/axes.)