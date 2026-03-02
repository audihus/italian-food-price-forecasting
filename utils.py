import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# =============================================================================
# Konstanta Global
# =============================================================================
HORIZON_OPTIONS = {'30 hari': 30, '90 hari': 90, '180 hari': 180}
CATEGORIES = ['Fruit', 'Vegetable', 'Meat']

FEATURES = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7',
            'rolling_mean_7',
            'day_of_week', 'day_of_month', 'month', 'quarter', 'season']

SEASON_MAP = {
    12: 0, 1: 0, 2: 0,
     3: 1, 4: 1, 5: 1,
     6: 2, 7: 2, 8: 2,
     9: 3, 10: 3, 11: 3
}

CATEGORY_MODEL_MAP = {
    'Fruit':     ('models/model_fruit.pkl',     'models/scaler_fruit.pkl'),
    'Vegetable': ('models/model_vegetable.pkl', 'models/scaler_vegetable.pkl'),
    'Meat':      ('models/model_meat.pkl',      'models/scaler_meat.pkl'),
}

CATEGORY_COLORS = {
    'Fruit':     '#2ecc71',
    'Vegetable': '#27ae60',
    'Meat':      '#e74c3c',
}

# =============================================================================
# Fungsi Utilitas
# =============================================================================

@st.cache_data  # ← CSV hanya dibaca sekali, hasil di-cache
def load_and_aggregate(csv_path: str, category: str) -> pd.DataFrame:
    """Load CSV, filter kategori, agregasi harian."""
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])

    df_daily = (
        df[df['COICOP4'] == category]
        .groupby(pd.Grouper(key='date', freq='D'))['price']
        .mean()
        .reset_index()
        .rename(columns={'date': 'day', 'price': 'avg_price'})
    )

    # Isi gap harian dengan forward fill
    df_daily = df_daily.set_index('day').asfreq('D').ffill().reset_index()
    return df_daily


def create_features(data: pd.DataFrame, lags: list = [1, 2, 3, 4, 5, 6, 7]) -> pd.DataFrame:
    """Membuat lag, rolling mean, dan calendar features."""
    df = data.copy()
    for lag in lags:
        df[f'lag_{lag}'] = df['avg_price'].shift(lag)
    df['rolling_mean_7'] = df['avg_price'].shift(1).rolling(window=7).mean()
    df['day_of_week']  = df['day'].dt.dayofweek
    df['day_of_month'] = df['day'].dt.day
    df['month']        = df['day'].dt.month
    df['quarter']      = df['day'].dt.quarter
    df['season']       = df['month'].map(SEASON_MAP)
    df = df.dropna().reset_index(drop=True)
    return df


def run_forecast(df_cat: pd.DataFrame, model, scaler, horizon: int) -> pd.DataFrame:
    """
    Forecast ke depan menggunakan model yang sudah di-load (tanpa re-fit).
    Menggunakan seluruh data historis untuk membangun history lag.
    """
    df = df_cat.copy()
    df['avg_price'] = scaler.transform(df[['avg_price']])
    df_feat = create_features(df)

    last_date = df_cat['day'].max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon,
        freq='D'
    )

    history = df_feat['avg_price'].tolist()
    preds = []

    for future_date in future_dates:
        row = {
            'lag_1': history[-1],
            'lag_2': history[-2],
            'lag_3': history[-3],
            'lag_4': history[-4],
            'lag_5': history[-5],
            'lag_6': history[-6],
            'lag_7': history[-7],
            'rolling_mean_7': np.mean(history[-7:]),
            'day_of_week':    future_date.dayofweek,
            'day_of_month':   future_date.day,
            'month':          future_date.month,
            'quarter':        future_date.quarter,
            'season':         SEASON_MAP[future_date.month],
        }
        X = np.array([row[f] for f in FEATURES]).reshape(1, -1)
        pred = model.predict(X)[0]
        preds.append(pred)
        history.append(pred)

    preds_actual = scaler.inverse_transform(
        np.array(preds).reshape(-1, 1)
    ).flatten()

    return pd.DataFrame({
        'Tanggal':  future_dates,
        'Forecast': preds_actual.round(4)
    })


def compute_summary(df_forecast: pd.DataFrame, df_hist: pd.DataFrame) -> dict:
    """Hitung ringkasan statistik forecast."""
    last_actual = df_hist['avg_price'].iloc[-1]
    first_fc    = df_forecast['Forecast'].iloc[0]
    last_fc     = df_forecast['Forecast'].iloc[-1]
    change_pct  = ((last_fc - last_actual) / last_actual) * 100

    return {
        'Harga Terakhir (Aktual)': f'€{last_actual:.4f}',
        'Forecast Awal':           f'€{first_fc:.4f}',
        'Forecast Akhir':          f'€{last_fc:.4f}',
        'Perubahan (%)':           f'{change_pct:+.2f}%',
        'Min Forecast':            f'€{df_forecast["Forecast"].min():.4f}',
        'Max Forecast':            f'€{df_forecast["Forecast"].max():.4f}',
    }