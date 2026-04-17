import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime, timedelta

# ====================== CLASS ENVIRONMENT (copy dari Phase 3) ======================
class CocoaFarmEnv(gym.Env):
    def __init__(self, model_profit, feature_cols, df_clean=None, max_steps=24):
        super().__init__()
        self.model_profit = model_profit
        self.feature_cols = feature_cols
        self.max_steps = max_steps
        
        # Action space (72 kombinasi)
        self.wilayah_list = ['Bali', 'Jawa Barat', 'Jawa Timur', 'NTT', 'Sulawesi', 'Sumatera']
        self.perkebunan_list = ['Perkebunan Negara', 'Perkebunan Rakyat', 'Perkebunan Swasta']
        self.proses_list = ['Fermentasi', 'Pengeringan', 'Penggilingan', 'Roasting']
        
        self.actions = [(w, p, pr) for w in self.wilayah_list 
                        for p in self.perkebunan_list for pr in self.proses_list]
        
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        
        self.current_step = 0
        self.last_profit = 0.0
        self.start_year = 2026
        self.start_month = 1
    
    def _get_obs(self):
        month = (self.current_step % 12) + 1
        sin = np.sin(2 * np.pi * month / 12)
        cos = np.cos(2 * np.pi * month / 12)
        last_norm = self.last_profit / 100_000_000
        return np.array([sin, cos, last_norm], dtype=np.float32)
    
    def reset(self, seed=None):
        self.current_step = 0
        self.last_profit = 0.0
        return self._get_obs(), {}
    
    def step(self, action):
        wilayah, jenis_perkebunan, jenis_proses = self.actions[action]
        month = ((self.current_step + self.start_month - 1) % 12) + 1
        year = self.start_year + ((self.current_step + self.start_month - 1) // 12)
        
        row_dict = {
            'Wilayah': wilayah,
            'Jenis_Perkebunan': jenis_perkebunan,
            'Jenis_Proses': jenis_proses,
            'Month': month,
            'Year': year,
            'Month_sin': np.sin(2 * np.pi * month / 12),
            'Month_cos': np.cos(2 * np.pi * month / 12),
        }
        
        row_df = pd.DataFrame([row_dict])
        encoded = pd.get_dummies(row_df, columns=['Wilayah', 'Jenis_Perkebunan', 'Jenis_Proses'], drop_first=True)
        encoded = encoded.reindex(columns=self.feature_cols, fill_value=0)
        
        pred_profit = max(0, self.model_profit.predict(encoded)[0])
        self.last_profit = pred_profit
        self.current_step += 1
        
        done = self.current_step >= self.max_steps
        return self._get_obs(), pred_profit, done, False, {'profit': pred_profit, 'action': (wilayah, jenis_perkebunan, jenis_proses)}

# ====================== STREAMLIT APP ======================
st.set_page_config(page_title="CocoaFarmRL Dashboard", layout="wide")
st.title("🌴 CocoaFarmRL – AI Agent untuk Optimalisasi Perkebunan Coklat")
st.markdown("**Agent RL sudah dilatih untuk memaksimalkan profit 2 tahun ke depan**")

# Load Models & Data
@st.cache_resource
def load_models():
    import os
    import warnings
    
    try:
        model_profit_rl = joblib.load("xgb_profit_rl.pkl")
        feature_cols_rl = joblib.load("feature_cols_rl.pkl")
        
        # Load PPO model with warning suppression for compatibility issues
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ppo_model = PPO.load("ppo_cocoafarm_agent")
        
        return model_profit_rl, feature_cols_rl, ppo_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

try:
    model_profit_rl, feature_cols_rl, ppo_model = load_models()
except Exception as e:
    st.error(f"Failed to initialize models: {str(e)}")
    st.stop()

# Sidebar
st.sidebar.header("🎛️ Pengaturan Simulasi")
start_date = st.sidebar.date_input("Tanggal Mulai Simulasi", datetime(2026, 1, 1))
months_to_simulate = st.sidebar.slider("Jumlah Bulan Simulasi", 12, 24, 24)
run_button = st.sidebar.button("🚀 Jalankan Simulasi dengan Agent RL", type="primary")

# Load environment
@st.cache_resource
def load_env():
    return CocoaFarmEnv(model_profit_rl, feature_cols_rl)

env = load_env()

if run_button:
    env.start_year = start_date.year
    env.start_month = start_date.month
    env.max_steps = months_to_simulate
    
    # Simulasi Agent
    obs, _ = env.reset()
    total_profit_agent = 0
    history = []
    
    for step in range(months_to_simulate):
        action, _ = ppo_model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        total_profit_agent += reward
        
        wilayah, perkebunan, proses = info['action']
        history.append({
            'Bulan': step + 1,
            'Wilayah': wilayah,
            'Jenis_Perkebunan': perkebunan,
            'Jenis_Proses': proses,
            'Profit_Bulanan': reward
        })
    
    df_history = pd.DataFrame(history)
    
    # Simulasi Random untuk perbandingan
    random_profits = []
    for _ in range(30):
        obs, _ = env.reset()
        total = 0
        for __ in range(months_to_simulate):
            action = env.action_space.sample()
            obs, reward, _, _, _ = env.step(action)
            total += reward
        random_profits.append(total)
    
    avg_random = np.mean(random_profits)
    
    # Tampilkan hasil
    col1, col2, col3 = st.columns(3)
    improvement = (((total_profit_agent/avg_random)-1)*100) if avg_random > 0 else 0
    col1.metric("Total Profit Agent RL", f"Rp {total_profit_agent:,.0f}", f"+{improvement:.1f}% vs Random")
    col2.metric("Rata-rata Profit Random", f"Rp {avg_random:,.0f}")
    col3.metric("Profit per Bulan (rata-rata)", f"Rp {total_profit_agent/months_to_simulate:,.0f}")
    
    st.subheader("📅 Rekomendasi Agent per Bulan")
    st.dataframe(df_history.style.format({"Profit_Bulanan": "Rp {:,.0f}"}), width='stretch')
    
    # Grafik
    fig_cum = px.line(df_history, x='Bulan', y=df_history['Profit_Bulanan'].cumsum(), 
                      title="Cumulative Profit Selama Simulasi", markers=True)
    st.plotly_chart(fig_cum, width='stretch')
    
    st.subheader("🔥 Distribusi Rekomendasi Agent")
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(px.pie(df_history, names='Wilayah', title="Wilayah Terpilih"), width='stretch')
    with col_b:
        st.plotly_chart(px.pie(df_history, names='Jenis_Proses', title="Proses Terpilih"), width='stretch')
    
    st.success("✅ Simulasi selesai! Agent RL sudah memberikan rekomendasi optimal.")
else:
    st.info("👈 Pilih tanggal dan klik tombol di sidebar untuk menjalankan simulasi")

st.caption("Dibuat dengan ❤️ untuk proyek Reinforcement Learning Perkebunan Coklat")
