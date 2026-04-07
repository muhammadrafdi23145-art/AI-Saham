import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.subplots as plt
import seaborn as sns
import plotly.graph_objects as go
import warnings

import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')

# --- KONFIGURASI HALAMAN STREAMLIT ---
st.set_page_config(page_title="AI Trading Forensik", layout="wide")
st.title("🤖 AI Trading Engine - Clean Data Edition")
st.markdown("Dashboard backtest berbasis Deep Learning (LSTM) dengan filter multikolinearitas.")

# --- SIDEBAR: INPUT INTERAKTIF ---
st.sidebar.header("⚙️ Parameter AI & Trading")

START_DATE = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
END_DATE = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))

st.sidebar.subheader("Hyperparameter AI")
WINDOW_SIZE = st.sidebar.slider("Window Size (Lag)", 5, 30, 10)
MIN_CONFIDENCE = st.sidebar.slider("Minimal Confidence AI", 0.50, 0.90, 0.55, 0.01)

st.sidebar.subheader("Manajemen Risiko")
INITIAL_CAPITAL = st.sidebar.number_input("Modal Awal (Rp)", min_value=1000000, value=100000000, step=1000000)
RISK_PER_TRADE = st.sidebar.slider("Alokasi per Trade", 0.1, 1.0, 0.2, 0.1)
STOP_LOSS = st.sidebar.number_input("Stop Loss (%)", value=3.0, step=0.5) / 100
TAKE_PROFIT = st.sidebar.number_input("Take Profit (%)", value=5.0, step=0.5) / 100

st.sidebar.subheader("Pilih Saham")
DAFTAR_SAHAM_TAYANG = [
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "TLKM.JK", 
    "ISAT.JK", "UNVR.JK", "ICBP.JK", "MYOR.JK", "ASII.JK", 
    "ADRO.JK", "PTBA.JK", "PGAS.JK", "KLBF.JK", "MIKA.JK", 
    "AMRT.JK", "MAPI.JK", "JSMR.JK", "SMGR.JK", "INTP.JK"
]
selected_stocks = st.sidebar.multiselect("Saham untuk Simulasi", DAFTAR_SAHAM_TAYANG, default=["JSMR.JK", "PTBA.JK", "MYOR.JK"])
SAHAM_ANALISIS = st.sidebar.selectbox("Fokus Analisis Chart & Heatmap", selected_stocks)

# Parameter Statis
THRESHOLD_RETURN = 0.02
COOLDOWN_DAYS = 3
MIN_DATA = 400
MIN_VOLATILITY = 0.015
FEE_BROKER = 0.0015

# --- UTILITAS ---
def fix_yfinance(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# --- TOMBOL EKSEKUSI ---
if st.sidebar.button("🚀 Jalankan AI Backtest", type="primary"):
    if not selected_stocks:
        st.warning("Pilih minimal 1 saham terlebih dahulu!")
        st.stop()
        
    hasil_backtest = {}
    data_chart = {}

    bar = st.progress(0, text="Mengunduh data makro...")
    df_ihsg = fix_yfinance(yf.download("^JKSE", start=START_DATE, end=END_DATE, progress=False))['Close']

    for idx, ticker in enumerate(selected_stocks):
        bar.progress((idx + 1) / len(selected_stocks), text=f"Memproses AI untuk {ticker}...")
        try:
            K.clear_session()
            df = fix_yfinance(yf.download(ticker, start=START_DATE, end=END_DATE, progress=False))
            if len(df) < MIN_DATA: continue

            df = df[['Open','High','Low','Close','Volume']].copy()
            df = df.join(df_ihsg.rename("IHSG"), how='left').ffill().bfill()

            # --- FEATURE ENGINEERING MURNI (TANPA PANDAS-TA) ---
            df['Ret_Close'] = df['Close'].pct_change()
            df['Ret_IHSG'] = df['IHSG'].pct_change()
            
            # EMA 50
            df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
            df['Trend'] = (df['Close'] > df['EMA_50']).astype(int)
            
            # ROC 10
            df['ROC_10'] = df['Close'].pct_change(periods=10) * 100

            # Bollinger Bands 20
            sma20 = df['Close'].rolling(window=20).mean()
            std20 = df['Close'].rolling(window=20).std()
            df['BBL'] = sma20 - (2 * std20)
            df['BBU'] = sma20 + (2 * std20)
            df['BB_width'] = (df['BBU'] - df['BBL']) / df['Close']

            if df['BB_width'].mean() < MIN_VOLATILITY: continue

            # Volume Price Trend (VPT)
            df['VPT'] = (df['Volume'] * ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1))).cumsum()
            
            # Target (1 = Naik > 2% besok)
            df['Target'] = (df['Close'].shift(-1) > df['Close'] * (1 + THRESHOLD_RETURN)).astype(int)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)

            features = ['Ret_Close', 'Ret_IHSG', 'ROC_10', 'BB_width', 'VPT']

            if ticker == SAHAM_ANALISIS:
                data_chart['correlation'] = df[features + ['Target']].corr()

            X = df[features].values
            y = df['Target'].values

            train_size = int(len(X) * 0.8)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[:train_size])
            X_test = scaler.transform(X[train_size:])

            X_train, y_train = create_sequences(X_train, y[:train_size], WINDOW_SIZE)
            X_test, y_test = create_sequences(X_test, y[train_size:], WINDOW_SIZE)

            if len(X_train) == 0: continue

            classes = np.unique(y_train)
            weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weight_dict = dict(zip(classes, weights))

            model = Sequential([
                LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2])),
                Dropout(0.3),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy')

            model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.15,
                      callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
                      class_weight=class_weight_dict, verbose=0)

            preds = model.predict(X_test, verbose=0)
            prob_buy_threshold = np.percentile(preds, 85)

            cash, saham, harga_beli, cooldown, total_trade = INITIAL_CAPITAL, 0, 0, 0, 0
            idx_test_start = train_size + WINDOW_SIZE
            dates_test = df.index[idx_test_start:]
            harga_open, harga_high, harga_low, harga_close, trend = df['Open'].iloc[idx_test_start:].values, df['High'].iloc[idx_test_start:].values, df['Low'].iloc[idx_test_start:].values, df['Close'].iloc[idx_test_start:].values, df['Trend'].iloc[idx_test_start:].values

            buy_markers, sell_markers = [], []

            for i in range(len(preds)):
                if cooldown > 0:
                    cooldown -= 1
                    continue

                if saham > 0:
                    if (harga_low[i] - harga_beli) / harga_beli <= -STOP_LOSS:
                        cash += (saham * harga_beli * (1 - STOP_LOSS)) * (1 - FEE_BROKER)
                        saham, cooldown = 0, COOLDOWN_DAYS
                        sell_markers.append((dates_test[i], harga_low[i], 'SL'))
                    elif (harga_high[i] - harga_beli) / harga_beli >= TAKE_PROFIT:
                        cash += (saham * harga_beli * (1 + TAKE_PROFIT)) * (1 - FEE_BROKER)
                        saham, cooldown = 0, COOLDOWN_DAYS
                        sell_markers.append((dates_test[i], harga_high[i], 'TP'))

                if preds[i][0] >= prob_buy_threshold and preds[i][0] >= MIN_CONFIDENCE and trend[i] == 1 and saham == 0:
                    qty = int((cash * RISK_PER_TRADE) // (harga_open[i] * (1 + FEE_BROKER)))
                    if qty > 0:
                        cash -= qty * harga_open[i] * (1 + FEE_BROKER)
                        saham, harga_beli = qty, harga_open[i]
                        total_trade += 1
                        buy_markers.append((dates_test[i], harga_open[i]))
                elif preds[i][0] <= 0.4 and saham > 0:
                    cash += (saham * harga_open[i]) * (1 - FEE_BROKER)
                    saham = 0
                    sell_markers.append((dates_test[i], harga_open[i], 'Exit'))

            if saham > 0: cash += (saham * harga_close[-1]) * (1 - FEE_BROKER)

            profit = ((cash - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
            hasil_backtest[ticker] = {'Profit (%)': profit, 'Total Trade': total_trade}

            if ticker == SAHAM_ANALISIS:
                data_chart['df'], data_chart['buy'], data_chart['sell'] = df.loc[dates_test], buy_markers, sell_markers

        except Exception as e:
            st.error(f"Error pada {ticker}: {e}")

    bar.empty()
    st.success("Backtest Selesai!")

    # --- TAMPILAN HASIL ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📊 Performa Portfolio")
        if hasil_backtest:
            df_hasil = pd.DataFrame(hasil_backtest).T
            alokasi_bobot = 100 / len(df_hasil)
            df_hasil['Alokasi Modal (Rp)'] = INITIAL_CAPITAL * (alokasi_bobot / 100)
            df_hasil['Hasil Akhir (Rp)'] = df_hasil['Alokasi Modal (Rp)'] * (1 + df_hasil['Profit (%)'] / 100)
            
            total_hasil = df_hasil['Hasil Akhir (Rp)'].sum()
            roi_total = ((total_hasil - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
            
            st.metric(label="Total Modal", value=f"Rp {INITIAL_CAPITAL:,.0f}")
            st.metric(label="Total Akhir", value=f"Rp {total_hasil:,.0f}", delta=f"{roi_total:.2f}%")
            
            st.dataframe(df_hasil[['Profit (%)', 'Total Trade']].style.format({'Profit (%)': '{:.2f}%'}))
        else:
            st.warning("Tidak ada saham yang lolos filter atau berhasil diproses.")

    with col2:
        if 'correlation' in data_chart:
            st.subheader(f"🧩 Heatmap Data Bersih: {SAHAM_ANALISIS}")
            fig_heat, ax_heat = plt.subplots(figsize=(8, 5))
            import seaborn as sns
            sns.heatmap(data_chart['correlation'], annot=True, cmap='coolwarm', fmt=".2f", ax=ax_heat)
            st.pyplot(fig_heat)

    st.markdown("---")
    
    if 'df' in data_chart:
        st.subheader(f"📈 Grafik Trading Interaktif: {SAHAM_ANALISIS}")
        df_plot = data_chart['df']
        
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Close'], mode='lines', name='Close Price', line=dict(color='black', width=1.5)))
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['EMA_50'], mode='lines', name='EMA 50', line=dict(color='blue', width=1, dash='dot')))
        
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['BBU'], mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['BBL'], mode='lines', fill='tonexty', fillcolor='rgba(128,128,128,0.2)', line=dict(width=0), name='Bollinger Bands'))

        if data_chart['buy']:
            buy_dates, buy_prices = zip(*data_chart['buy'])
            fig.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode='markers', name='AI Buy', marker=dict(color='green', symbol='triangle-up', size=12, line=dict(color='DarkGreen', width=1))))

        if data_chart['sell']:
            sell_dates, sell_prices, sell_types = zip(*data_chart['sell'])
            hover_text = [f"Type: {t}<br>Price: {p}" for t, p in zip(sell_types, sell_prices)]
            fig.add_trace(go.Scatter(x=sell_dates, y=sell_prices, mode='markers', name='AI Sell', text=hover_text, hoverinfo='text', marker=dict(color='red', symbol='triangle-down', size=12, line=dict(color='DarkRed', width=1))))

        fig.update_layout(title=f"Analisis Jejak Out-of-Sample {SAHAM_ANALISIS}", xaxis_title='Tanggal', yaxis_title='Harga', template='plotly_white', hovermode='x unified', height=600)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("👈 Atur parameter di sidebar dan klik **'Jalankan AI Backtest'** untuk memulai simulasi.")
