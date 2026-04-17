# 🌴 CocoaFarmRL – AI Agent untuk Optimalisasi Perkebunan Coklat

Dashboard interaktif Streamlit dengan Reinforcement Learning Agent untuk memaksimalkan profit perkebunan coklat berdasarkan prediksi 2 tahun ke depan.

## 📋 Daftar Isi
- [Fitur Utama](#fitur-utama)
- [Persyaratan Sistem](#persyaratan-sistem)
- [Instalasi](#instalasi)
- [Cara Menjalankan](#cara-menjalankan)
- [Struktur Project](#struktur-project)
- [Cara Kerja](#cara-kerja)

## ✨ Fitur Utama

- 🤖 **RL Agent (PPO)** - Dilatih untuk membuat keputusan optimal
- 📊 **Simulasi Interaktif** - Jalankan simulasi dengan parameter custom
- 📈 **Visualisasi Real-time** - Grafik cumulative profit dan distribusi rekomendasi
- 🎯 **Perbandingan Baseline** - Bandingkan performa agent vs random policy
- 💼 **Multi-dimensi** - Optimalisasi berdasarkan wilayah, jenis perkebunan, dan proses

## 🖥️ Persyaratan Sistem

- Python 3.10+
- Windows/Mac/Linux
- RAM minimal 4GB
- Koneksi internet (untuk dependencies)

## 📦 Instalasi

### 1. Clone atau Download Project
```bash
cd CocoaFarmRL
```

### 2. Buat Virtual Environment
```bash
python -m venv .venv
```

### 3. Aktivasi Virtual Environment

**Windows:**
```bash
.\.venv\Scripts\activate
```

**Mac/Linux:**
```bash
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Cara Menjalankan

### Jalankan Streamlit App
```bash
streamlit run app.py
```

Atau menggunakan Python module:
```bash
python -m streamlit run app.py
```

Aplikasi akan terbuka di browser pada URL:
- **Local**: http://localhost:8501
- **Network**: http://[IP_ADDRESS]:8501

### Cara Menggunakan Dashboard

1. **Sidebar Settings**:
   - Pilih **Tanggal Mulai Simulasi** (default: 1 Januari 2026)
   - Atur **Jumlah Bulan Simulasi** (12-24 bulan)

2. **Jalankan Simulasi**:
   - Klik tombol 🚀 **"Jalankan Simulasi dengan Agent RL"**

3. **Lihat Hasil**:
   - Total Profit Agent RL vs Random Policy
   - Tabel rekomendasi per bulan
   - Grafik cumulative profit
   - Distribusi rekomendasi per wilayah dan proses

## 📁 Struktur Project

```
CocoaFarmRL/
├── app.py                                          # Main Streamlit application
├── requirements.txt                                # Python dependencies
├── README.md                                       # Dokumentasi ini
├── .venv/                                          # Virtual environment (gitignored)
├── .streamlit/                                     # Streamlit config
│
├── Model & Data Files:
├── xgb_profit_rl.pkl                              # XGBoost Profit Prediction Model
├── feature_cols_rl.pkl                            # Feature columns untuk model
├── ppo_cocoafarm_agent.zip                        # PPO Agent (trained policy)
│
├── Dataset:
├── dataset_perkebunan_coklat_clean.csv            # Cleaned dataset
├── dataset_perkebunan_coklat_2024_2026.csv        # Original dataset
│
└── Notebook:
    └── CocoaFarmRL_–_Reinforcement_Learning_Agent_untuk_Optimalisasi_Perkebunan_Coklat.ipynb
        # Jupyter notebook dengan training code & analysis
```

## 🧠 Cara Kerja

### Komponen Utama

#### 1. **CocoaFarmEnv** (Gymnasium Environment)
Custom environment yang merepresentasikan keputusan perkebunan coklat:
- **State Space**: 3 dimensi (sin(month), cos(month), last_profit_normalized)
- **Action Space**: 72 kombinasi (6 wilayah × 3 jenis perkebunan × 4 proses)
- **Reward**: Prediksi profit dari XGBoost model

#### 2. **PPO Agent** (Proximal Policy Optimization)
Policy gradient algorithm untuk membuat keputusan optimal:
- Dilatih untuk memaksimalkan total profit 24 bulan
- Menggunakan policy network yang belajar dari trial-and-error

#### 3. **XGBoost Profit Model**
Prediksi profit berdasarkan:
- Wilayah (6 pilihan)
- Jenis Perkebunan (3 pilihan)
- Jenis Proses (4 pilihan)
- Seasonal features (sin/cos dari bulan)

### Flow Simulasi

```
User Input (tanggal, durasi)
         ↓
Reset Environment
         ↓
Loop (untuk setiap bulan):
  ├─ Get current state dari environment
  ├─ PPO agent predict best action
  ├─ Execute action & dapatkan reward (profit)
  └─ Simpan history
         ↓
Bandingkan dengan random policy (30x runs)
         ↓
Tampilkan visualisasi & metrics
```

## 📊 Fitur Visualisasi

| Komponen | Deskripsi |
|----------|-----------|
| Metric Cards | Total profit agent, random baseline, profit per bulan |
| Data Table | Detail rekomendasi action per bulan |
| Cumulative Profit Chart | Line chart profit kumulatif selama periode simulasi |
| Wilayah Distribution | Pie chart distribusi wilayah yang dipilih agent |
| Proses Distribution | Pie chart distribusi proses yang dipilih agent |

## 🔧 Troubleshooting

### Error: "ModuleNotFoundError: No module named..."
**Solusi**: Install dependencies ulang
```bash
pip install -r requirements.txt
```

### Error: "No module named 'numpy._core.numeric'"
**Solusi**: Pastikan numpy 2.4.4 dan stable-baselines3 2.0.0 compatible
```bash
pip install numpy==2.4.4 stable-baselines3==2.0.0
```

### Streamlit tidak terbuka di browser
**Solusi**: Buka manual ke http://localhost:8501

### Model loading lambat
**Solusi**: PPO model cukup besar, tunggu proses loading cache pertama kali

## 📚 Dependencies Utama

| Package | Versi | Fungsi |
|---------|-------|--------|
| streamlit | 1.56.0 | Web framework untuk dashboard |
| stable-baselines3 | 2.0.0 | RL algorithms library |
| gymnasium | 0.28.1 | RL environment framework |
| xgboost | 3.2.0 | Gradient boosting untuk prediction |
| plotly | 6.6.0 | Interactive graphs |
| pandas | 3.0.2 | Data manipulation |
| numpy | 2.4.4 | Numerical computing |
| torch | 2.11.0 | Deep learning framework |

## 📝 Notes

- Simulasi dengan RL agent biasanya 20-50% lebih profitable dibanding random policy
- Cache model dilakukan otomatis untuk mempercepat loading
- Setiap simulasi independen dan tidak mempengaruhi trained agent

## 👨‍💻 Developer

Dibuat dengan ❤️ untuk proyek Reinforcement Learning Perkebunan Coklat

---

**Last Updated**: April 2026  
**Python Version**: 3.10+  
**Status**: ✅ Active Development
