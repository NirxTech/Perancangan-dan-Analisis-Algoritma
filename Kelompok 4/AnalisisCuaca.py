import sys

try:
    import argparse
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import cross_val_score
    import warnings
    import os
except Exception as e:
    missing = None
    if isinstance(e, ModuleNotFoundError):
        missing = e.name
    print("ERROR: Modul Python yang diperlukan tidak ditemukan.")
    if missing:
        print(f"   Modul hilang: {missing}")
    print("Solusi:")
    print("  1) Pasang seluruh dependensi dari requirements.txt (direkomendasikan):")
    print('     py -m pip install -r "Kelompok 4\\requirements.txt"')
    print("  2) Atau pasang modul yang hilang saja, mis.:")
    print("     py -m pip install pandas numpy matplotlib seaborn statsmodels scikit-learn openpyxl")
    print("Setelah instalasi, jalankan ulang script.")
    sys.exit(1)

warnings.filterwarnings('ignore')

# ==========================================
# 1. MEMBACA DATASET & PREPROCESSING (.csv)
# ==========================================
# Argparse untuk input file dan horizon prediksi
parser = argparse.ArgumentParser(description='Analisis Curah Hujan: Holt-Winters, Decision Tree, Linear Regression')
parser.add_argument('file', nargs='?', default='Dataset_Curah_Hujan_Kota_Padang/dataset_curah_hujan_teluk_bayur_2024_2025.csv', help='Nama file CSV dataset (default: Dataset_Curah_Hujan_Kota_Padang/dataset_curah_hujan_teluk_bayur_2024_2025.csv)')
parser.add_argument('-H', '--horizon', type=int, default=12, help='Horizon prediksi dalam bulan (mis. 6, 12)')
args = parser.parse_args()
file_name = args.file
FORECAST_HORIZON = max(1, int(args.horizon))

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))

    def resolve_input_path(fname):
        # If absolute path provided, accept if exists
        if os.path.isabs(fname):
            return fname if os.path.isfile(fname) else None

        # Try several sensible relative locations
        candidates = [
            fname,
            os.path.join(script_dir, fname),
            os.path.join(os.getcwd(), fname),
            os.path.join(script_dir, os.pardir, fname),
        ]
        for c in candidates:
            c_norm = os.path.normpath(c)
            if os.path.isfile(c_norm):
                return c_norm

        # Walk upwards from script_dir to root, trying the relative path at each level
        cur = script_dir
        while True:
            try_path = os.path.normpath(os.path.join(cur, fname))
            if os.path.isfile(try_path):
                return try_path
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent

        # If still not found, try to locate a file with the same basename under the project
        target_basename = os.path.basename(fname)
        search_root = os.path.abspath(os.path.join(script_dir, os.pardir))
        for root, dirs, files in os.walk(search_root):
            if target_basename in files:
                return os.path.join(root, target_basename)

        # Final fallback: search cwd tree
        for root, dirs, files in os.walk(os.getcwd()):
            if target_basename in files:
                return os.path.join(root, target_basename)

        return None

    if not os.path.isabs(file_name):
        resolved = resolve_input_path(file_name)
        if resolved:
            file_name = resolved
        else:
            raise FileNotFoundError(file_name)

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

    # 1) Load data dari satu file CSV
    df_combined = load_csv_timeseries(file_name)
    if df_combined is None or df_combined.empty:
        raise ValueError(f"Data kosong atau format CSV tidak valid: {file_name}")

    # 2) Agregasi mean per waktu
    df_combined['Bulan_Tahun'] = pd.to_datetime(df_combined['Bulan_Tahun'], errors='coerce')
    df_combined['Curah_Hujan'] = pd.to_numeric(df_combined['Curah_Hujan'], errors='coerce')
    df_combined = df_combined.dropna(subset=['Bulan_Tahun', 'Curah_Hujan'])
    df_combined = df_combined.groupby('Bulan_Tahun', as_index=False)['Curah_Hujan'].mean().sort_values('Bulan_Tahun').reset_index(drop=True)

    # Finalisasi dataframe yang siap pakai untuk algoritma di bawah
    df = df_combined
    df['Time_Index'] = np.arange(1, len(df) + 1)

    print("Data CSV berhasil dimuat dan diagregasi!\n")
    # Ringkasan singkat agar output lebih rapi
    try:
        print(f"Jumlah observasi setelah preprocessing: {len(df)}")
        if len(df) > 0:
            print(f"Rentang tanggal: {df['Bulan_Tahun'].min().strftime('%Y-%m')} sampai {df['Bulan_Tahun'].max().strftime('%Y-%m')}")
            print('\nContoh data (10 baris pertama):')
            print(df.head(10).to_string(index=False))
    except Exception:
        pass

    # Jika hasil preprocessing kosong, tampilkan informasi diagnostik dan hentikan program
    if df.empty:
        print("Hasil preprocessing menghasilkan dataframe kosong. Periksa format file CSV Kamu.")
        try:
            print('\n--- Cuplikan data mentah (df_raw.head()) ---')
            print(df_combined.head().to_string(index=False))
        except Exception:
            pass
        print('\nPetunjuk: Pastikan CSV memiliki kolom tanggal/waktu dan curah hujan.')
        print('Jika format CSV berbeda, sesuaikan nama kolom waktu/nilai pada helper loader.')
        exit()

except FileNotFoundError:
    print(f"ERROR: File '{file_name}' tidak ditemukan.")
    print("Pastikan file .csv berada di dalam folder yang sama dengan script ini.")
    exit()
except ImportError:
    print("ERROR: Library yang dibutuhkan belum terinstall.")
    exit()
except Exception as e:
    print(f"Terjadi kesalahan saat memproses data: {e}")
    exit()

# ==========================================
# 2. ALGORITMA 1: HOLT-WINTERS (PREDIKSI)
# ==========================================
print("="*50)
print("ALGORITMA 1: HOLT-WINTERS (PREDIKSI)")
print("="*50)

# Siapkan series dengan index datetime
series = df.set_index('Bulan_Tahun')['Curah_Hujan']
n = len(series)

prediksi_hw = None
hw_method_desc = ''
hw_mae = None
hw_rmse = None

if n == 0:
    print("Data kosong — tidak ada yang bisa diproses untuk prediksi.")
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
            hw_mae = mean_absolute_error(series, fitted)
            hw_rmse = mean_squared_error(series, fitted, squared=False)
            print(f"\nMetrik in-sample Holt-Winters: MAE = {hw_mae:.3f}, RMSE = {hw_rmse:.3f}")
        except Exception:
            pass

    except Exception as e:
        print(f"Holt-Winters gagal: {e}")
        prediksi_hw = None


# ==========================================
# 3. ALGORITMA 2: DECISION TREE (KLASIFIKASI)
# ==========================================
print("\n" + "="*50)
print("ALGORITMA 2: DECISION TREE (KLASIFIKASI)")
print("="*50)

def compute_equal_interval_thresholds(series, k=3):
    """Hitung threshold untuk k kategori menggunakan metode equal-interval."""
    xmin = float(series.min())
    xmax = float(series.max())
    R = xmax - xmin
    I = R / k if k else 0
    low_max = xmin + I
    mid_max = xmin + 2 * I
    return xmin, xmax, low_max, mid_max, I

xmin, xmax, low_max, mid_max, interval_len = compute_equal_interval_thresholds(df['Curah_Hujan'], k=3)

def label_kategori_equal_interval(x):
    if x <= low_max:
        return 'Rendah'
    elif x <= mid_max:
        return 'Sedang'
    else:
        return 'Tinggi'

def kategori_ke_nilai(kategori):
    if kategori == 'Rendah':
        return (xmin + low_max) / 2
    if kategori == 'Sedang':
        return (low_max + mid_max) / 2
    return (mid_max + xmax) / 2


df['Kategori'] = df['Curah_Hujan'].apply(label_kategori_equal_interval)

X_dt = df[['Time_Index']]
y_dt = df['Kategori']
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_trained = False
dt_future_df = pd.DataFrame()
train_acc = None
cv_scores = None
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
        future_dt_idx = np.arange(len(df) + 1, len(df) + FORECAST_HORIZON + 1)
        future_dt_dates = pd.date_range(start=df['Bulan_Tahun'].iloc[-1] + pd.DateOffset(months=1), periods=FORECAST_HORIZON, freq='MS')
        future_dt_categories = dt_model.predict(pd.DataFrame({'Time_Index': future_dt_idx}))
        future_dt_values = [kategori_ke_nilai(cat) for cat in future_dt_categories]
        dt_future_df = pd.DataFrame({
            'Bulan_Tahun': future_dt_dates,
            'Kategori_Prediksi': future_dt_categories,
            'Prediksi_Curah_Hujan': future_dt_values,
        })
        print(f"\nPrediksi Decision Tree {FORECAST_HORIZON} bulan ke depan:")
        print(dt_future_df.to_string(index=False))
        try:
            dt_future_df.to_csv('prediksi_dt.csv', index=False)
            print('Prediksi Decision Tree disimpan di: prediksi_dt.csv')
        except Exception as e:
            print(f"Gagal menyimpan prediksi_dt.csv: {e}")
    except Exception:
        pass


# ==========================================
# 4. ALGORITMA 3: LINEAR REGRESSION (TREN)
# ==========================================
print("\n" + "="*50)
print("ALGORITMA 3: LINEAR REGRESSION (ANALISIS TREN)")
print("="*50)

X_lr = df[['Time_Index']]
y_lr = df['Curah_Hujan']
lr_pred_values = None
lr_future_df = pd.DataFrame()
r2 = None

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
            print(f"Kesimpulan Tren : MENINGKAT (Slope bernilai positif)")
        elif slope < 0:
            print(f"Kesimpulan Tren : MENURUN (Slope bernilai negatif)")
        else:
            print(f"Kesimpulan Tren : STABIL / FLAT (Slope = 0)")

        future_lr_idx = np.arange(len(df) + 1, len(df) + FORECAST_HORIZON + 1)
        future_lr_dates = pd.date_range(start=df['Bulan_Tahun'].iloc[-1] + pd.DateOffset(months=1), periods=FORECAST_HORIZON, freq='MS')
        future_lr_values = intercept + (slope * future_lr_idx)
        lr_future_df = pd.DataFrame({'Bulan_Tahun': future_lr_dates, 'Prediksi_Curah_Hujan': future_lr_values})
        print(f"\nPrediksi Linear Regression {FORECAST_HORIZON} bulan ke depan:")
        print(lr_future_df.to_string(index=False))
        try:
            lr_future_df.to_csv('prediksi_lr.csv', index=False)
            print('Prediksi Linear Regression disimpan di: prediksi_lr.csv')
        except Exception as e:
            print(f"Gagal menyimpan prediksi_lr.csv: {e}")

    except Exception as e:
        print(f"Linear Regression gagal: {e}")
else:
    print("Linear Regression membutuhkan setidaknya 2 observasi — dilewati.")
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

try:
    out_df = df.copy()
    out_df.to_csv('results_summary.csv', index=False)
    print("\nHasil ringkasan disimpan di: results_summary.csv")
except Exception as e:
    print(f"⚠️ Gagal menyimpan results_summary.csv: {e}")

pred_df = pd.DataFrame()
if prediksi_hw is not None:
    try:
        future_dates = pd.date_range(start=df['Bulan_Tahun'].iloc[-1] + pd.DateOffset(months=1), periods=FORECAST_HORIZON, freq='MS')
        pred_df = pd.DataFrame({'Bulan_Tahun': future_dates, 'Prediksi_Curah_Hujan': [max(0, float(v)) for v in prediksi_hw]})
        pred_df['Kategori'] = pred_df['Prediksi_Curah_Hujan'].apply(label_kategori_equal_interval)
        pred_df.to_csv('prediksi_hw.csv', index=False)
        print("Prediksi Holt-Winters disimpan di: prediksi_hw.csv")
    except Exception as e:
        print(f"Gagal menyimpan prediksi_hw.csv: {e}")

try:
    hist_monthly = series.groupby(series.index.month).mean()
    overall_mean = series.mean()
    if not pred_df.empty:
        print('\nPenjelasan prediksi per-bulan:')
        for _, row in pred_df.iterrows():
            m = row['Bulan_Tahun'].month
            pred_val = row['Prediksi_Curah_Hujan']
            hist_avg = hist_monthly.get(m, overall_mean)
            diff = pred_val - hist_avg
            direction = 'lebih tinggi' if diff > 0 else ('lebih rendah' if diff < 0 else 'sama dengan')
            print(f"- {row['Bulan_Tahun'].strftime('%Y-%m')}: prediksi {pred_val:.2f} mm; rata-rata historis bulan ini {hist_avg:.2f} mm -> {direction} sebesar {abs(diff):.2f} mm")
except Exception:
    pass

try:
    if not pred_df.empty:
        pred_df['Year'] = pred_df['Bulan_Tahun'].dt.year
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

def rolling_backtest(series, h, initial_train=None):
    n = len(series)
    if initial_train is None:
        initial_train = max(24, int(n * 0.7))
    if n < initial_train + h:
        return None
    preds = []
    trues = []
    i = initial_train
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
            trues.extend(list(series.iloc[i:i + h]))
        except Exception:
            break
        i += h
    if len(preds) == 0:
        return None
    mae = mean_absolute_error(trues, preds)
    rmse = mean_squared_error(trues, preds, squared=False)
    return {'mae': mae, 'rmse': rmse, 'points': len(preds)}


def hw_holdout_score(series, holdout_size):
    if len(series) < holdout_size + 3:
        return None
    train = series.iloc[:-holdout_size]
    test = series.iloc[-holdout_size:]
    try:
        if len(train) >= 24:
            model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12).fit()
        else:
            model = ExponentialSmoothing(train, trend='add', seasonal=None).fit()
        forecast = model.forecast(holdout_size)
        return mean_squared_error(test, forecast, squared=False)
    except Exception:
        return None

backtest_res = None
try:
    backtest_res = rolling_backtest(series, min(FORECAST_HORIZON, 6))
    if backtest_res is not None:
        print(f"\nHasil backtest (h={min(FORECAST_HORIZON,6)}): MAE={backtest_res['mae']:.3f}, RMSE={backtest_res['rmse']:.3f} (dari {backtest_res['points']} titik)")
    else:
        print('\nBacktest tidak dilakukan: data tidak mencukupi untuk evaluasi out-of-sample sederhana.')
except Exception:
    pass

def add_dataset_snapshot(fig, rows_df, title):
    fig.text(0.5, 0.02, title + '\n' + rows_df.to_string(index=False), ha='center', va='bottom', fontsize=8, family='monospace')

def save_figure(fig, path):
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(path)
    print(f"Plot disimpan di: {path}")

try:
    sns.set_style('whitegrid')
    out_dir = 'output_plots'
    os.makedirs(out_dir, exist_ok=True)

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
    print(f"Boxplot bulanan disimpan di: {boxfile}")

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
        print(f"Heatmap bulanan-tahunan disimpan di: {heatfile}")
    else:
        print("Heatmap dilewati: data pivot kosong.")

    monthly_avg = df.groupby(df['Bulan_Tahun'].dt.month)['Curah_Hujan'].mean().sort_index()
    fig_avg, ax_avg = plt.subplots(figsize=(10, 5))
    sns.barplot(x=monthly_avg.index.tolist(), y=monthly_avg.values.flatten(), palette='Blues', ax=ax_avg)
    ax_avg.set_title('Rata-rata Curah Hujan per Bulan (Historis)')
    ax_avg.set_xlabel('Bulan (1=Jan .. 12=Des)')
    ax_avg.set_ylabel('Curah Hujan rata-rata (mm)')
    avgfile = os.path.join(out_dir, 'monthly_average.png')
    fig_avg.tight_layout()
    fig_avg.savefig(avgfile)
    print(f"Grafik rata-rata bulanan disimpan di: {avgfile}")

    # Visual 1: Holt-Winters
    hw_fig = plt.figure(figsize=(14, 10))
    hw_gs = hw_fig.add_gridspec(2, 1, height_ratios=[4, 1])
    hw_ax = hw_fig.add_subplot(hw_gs[0])
    hw_tbl_ax = hw_fig.add_subplot(hw_gs[1])
    hw_ax.plot(df['Bulan_Tahun'], df['Curah_Hujan'], label='Data Aktual Historis', color='#1f77b4', marker='o', markersize=4)
    if not pred_df.empty:
        hw_ax.plot(pred_df['Bulan_Tahun'], pred_df['Prediksi_Curah_Hujan'], label=f'Prediksi Holt-Winters ({FORECAST_HORIZON} Bulan)', color='#d62728', linestyle='dashed', marker='x')
        for _, row in pred_df.iterrows():
            warna = {'Rendah': '#2ca02c', 'Sedang': '#ff7f0e', 'Tinggi': '#d62728'}.get(row['Kategori'], '#333333')
            hw_ax.scatter(row['Bulan_Tahun'], row['Prediksi_Curah_Hujan'], color=warna, s=55, zorder=5)
            hw_ax.annotate(row['Kategori'], (row['Bulan_Tahun'], row['Prediksi_Curah_Hujan']), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8, color=warna)
    hw_ax.set_title('Output Algoritma Holt-Winters', fontsize=14)
    hw_ax.set_xlabel('Tanggal')
    hw_ax.set_ylabel('Curah Hujan (mm)')
    hw_ax.legend(loc='upper left')
    hw_ax.grid(True, linestyle='--', alpha=0.6)
    hw_tbl_ax.axis('off')
    hw_tbl_ax.set_title('Snapshot Dataset')
    hw_tbl = df[['Bulan_Tahun', 'Curah_Hujan', 'Kategori']].tail(5).copy()
    hw_tbl['Bulan_Tahun'] = hw_tbl['Bulan_Tahun'].dt.strftime('%Y-%m')
    hw_tbl_ax.table(cellText=hw_tbl.values, colLabels=hw_tbl.columns, loc='center', cellLoc='center')
    hwfile = os.path.join(out_dir, 'holt_winters_output.png')
    save_figure(hw_fig, hwfile)

    # Visual 2: Decision Tree
    dt_fig = plt.figure(figsize=(18, 14))
    dt_gs = dt_fig.add_gridspec(3, 1, height_ratios=[3, 2, 1])
    dt_tree_ax = dt_fig.add_subplot(dt_gs[0])
    dt_pred_ax = dt_fig.add_subplot(dt_gs[1])
    dt_tbl_ax = dt_fig.add_subplot(dt_gs[2])
    if dt_trained:
        plot_tree(dt_model, feature_names=['Time_Index'], class_names=[str(c) for c in dt_model.classes_], filled=True, rounded=True, ax=dt_tree_ax, fontsize=8)
        dt_tree_ax.set_title('Visual Decision Tree (Pohon Keputusan)')
    else:
        dt_tree_ax.text(0.5, 0.5, 'Decision Tree tidak tersedia', ha='center', va='center')
        dt_tree_ax.set_axis_off()
    dt_pred_ax.plot(df['Bulan_Tahun'], df['Curah_Hujan'], label='Data Aktual Historis', color='#7f7f7f', alpha=0.75)
    if not dt_future_df.empty:
        dt_pred_ax.plot(dt_future_df['Bulan_Tahun'], dt_future_df['Prediksi_Curah_Hujan'], label='Prediksi Decision Tree', color='#2ca02c', marker='o')
        for _, row in dt_future_df.iterrows():
            dt_pred_ax.annotate(row['Kategori_Prediksi'], (row['Bulan_Tahun'], row['Prediksi_Curah_Hujan']), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8)
    dt_pred_ax.set_title('Prediksi Decision Tree terhadap Data Historis')
    dt_pred_ax.set_xlabel('Tanggal')
    dt_pred_ax.set_ylabel('Curah Hujan ekuivalen (mm)')
    dt_pred_ax.legend(loc='upper left')
    dt_pred_ax.grid(True, linestyle='--', alpha=0.6)
    dt_tbl_ax.axis('off')
    dt_tbl_ax.set_title('Snapshot Dataset')
    dt_tbl = df[['Bulan_Tahun', 'Curah_Hujan', 'Kategori']].tail(5).copy()
    dt_tbl['Bulan_Tahun'] = dt_tbl['Bulan_Tahun'].dt.strftime('%Y-%m')
    dt_tbl_ax.table(cellText=dt_tbl.values, colLabels=dt_tbl.columns, loc='center', cellLoc='center')
    dtfile = os.path.join(out_dir, 'decision_tree_output.png')
    save_figure(dt_fig, dtfile)

    # Visual 3: Linear Regression
    lr_fig = plt.figure(figsize=(14, 10))
    lr_gs = lr_fig.add_gridspec(2, 1, height_ratios=[4, 1])
    lr_ax = lr_fig.add_subplot(lr_gs[0])
    lr_tbl_ax = lr_fig.add_subplot(lr_gs[1])
    lr_ax.plot(df['Bulan_Tahun'], df['Curah_Hujan'], label='Data Aktual Historis', color='#7f7f7f', alpha=0.7)
    if lr_trained and lr_pred_values is not None:
        lr_ax.plot(df['Bulan_Tahun'], lr_pred_values, label=f'Garis Tren Historis (y = {intercept:.2f} + {slope:.4f}x)', color='#2ca02c', linewidth=3)
    if not lr_future_df.empty:
        lr_ax.plot(lr_future_df['Bulan_Tahun'], lr_future_df['Prediksi_Curah_Hujan'], label='Prediksi Linear Regression', color='#d62728', linestyle='dashed', marker='x')
    lr_ax.set_title('Output Algoritma Linear Regression', fontsize=14)
    lr_ax.set_xlabel('Tanggal')
    lr_ax.set_ylabel('Curah Hujan (mm)')
    lr_ax.legend(loc='upper left')
    lr_ax.grid(True, linestyle='--', alpha=0.6)
    lr_tbl_ax.axis('off')
    lr_tbl_ax.set_title('Snapshot Dataset')
    lr_tbl = df[['Bulan_Tahun', 'Curah_Hujan', 'Time_Index']].tail(5).copy()
    lr_tbl['Bulan_Tahun'] = lr_tbl['Bulan_Tahun'].dt.strftime('%Y-%m')
    lr_tbl_ax.table(cellText=lr_tbl.values, colLabels=lr_tbl.columns, loc='center', cellLoc='center')
    lrfile = os.path.join(out_dir, 'linear_regression_output.png')
    save_figure(lr_fig, lrfile)

    raw_scores = {}
    hw_error = backtest_res['rmse'] if backtest_res is not None else hw_rmse
    if hw_error is None:
        hw_error = hw_holdout_score(series, min(max(3, FORECAST_HORIZON), max(3, len(series) // 10)))
    if hw_error is None and prediksi_hw is not None and len(series) > 0:
        recent_actual = float(series.tail(min(3, len(series))).mean())
        recent_forecast = float(np.mean([max(0, float(v)) for v in prediksi_hw[:min(3, len(prediksi_hw))]]))
        hw_error = abs(recent_forecast - recent_actual)
    if hw_error is not None:
        data_range = max(xmax - xmin, 1e-9)
        raw_scores['Holt-Winters'] = max(0.0, 100.0 * (1.0 - (float(hw_error) / data_range)))
    if dt_trained:
        dt_score = float(cv_scores.mean()) if cv_scores is not None else float(train_acc if train_acc is not None else 0.0)
        raw_scores['Decision Tree'] = max(0.0, dt_score * 100.0)
    if lr_trained and r2 is not None:
        raw_scores['Linear Regression'] = max(0.0, float(r2) * 100.0)

    if len(raw_scores) == 3:
        total_raw = sum(raw_scores.values()) or 1.0
        comparison_percentages = {name: (score / total_raw) * 100.0 for name, score in raw_scores.items()}
        combined_percentage = sum(raw_scores.values()) / 3.0
        print('\nPersentase perbandingan algoritma:')
        for name, pct in comparison_percentages.items():
            print(f'- {name}: {pct:.2f}%')
        print(f'Persentase gabungan tiga algoritma: {combined_percentage:.2f}%')

        cmp_fig, (cmp_ax1, cmp_ax2) = plt.subplots(1, 2, figsize=(16, 7))
        bars = cmp_ax1.bar(list(raw_scores.keys()), list(raw_scores.values()), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        cmp_ax1.set_title('Perbandingan Nilai Persentase Algoritma')
        cmp_ax1.set_ylabel('Persentase performa (%)')
        cmp_ax1.set_ylim(0, 100)
        for bar, value in zip(bars, raw_scores.values()):
            cmp_ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{value:.1f}%', ha='center', va='bottom', fontsize=9)

        labels = list(comparison_percentages.keys()) + ['Gabungan']
        values = list(comparison_percentages.values()) + [combined_percentage]
        cmp_ax2.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd'])
        cmp_ax2.set_title('Persentase Gabungan dan Komposisi Relatif')
        cmpfile = os.path.join(out_dir, 'comparison_percentages.png')
        save_figure(cmp_fig, cmpfile)
    else:
        print('\nPersentase perbandingan belum dibuat karena salah satu algoritma belum menghasilkan skor.')

    main_fig, main_ax = plt.subplots(figsize=(14, 7))
    main_ax.plot(df['Bulan_Tahun'], df['Curah_Hujan'], label='Data Aktual Historis', color='#1f77b4')
    if not pred_df.empty:
        main_ax.plot(pred_df['Bulan_Tahun'], pred_df['Prediksi_Curah_Hujan'], label='Prediksi Holt-Winters', color='#d62728', linestyle='dashed')
    if not lr_future_df.empty:
        main_ax.plot(lr_future_df['Bulan_Tahun'], lr_future_df['Prediksi_Curah_Hujan'], label='Prediksi Linear Regression', color='#2ca02c', linestyle='dotted')
    if not dt_future_df.empty:
        main_ax.plot(dt_future_df['Bulan_Tahun'], dt_future_df['Prediksi_Curah_Hujan'], label='Prediksi Decision Tree', color='#9467bd', linestyle='dashdot')
    main_ax.set_title('Ringkasan Prediksi Curah Hujan')
    main_ax.set_xlabel('Tanggal')
    main_ax.set_ylabel('Curah Hujan (mm)')
    main_ax.legend(loc='upper left')
    main_ax.grid(True, linestyle='--', alpha=0.6)
    mainfile = os.path.join('output_plots', 'main_plot.png')
    save_figure(main_fig, mainfile)

    plt.show(block=True)
except Exception as e:
    print(f"Error saat membuat visualisasi tambahan: {e}")
