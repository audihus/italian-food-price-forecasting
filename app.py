import io
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils import (
    CATEGORY_COLORS, CATEGORY_MODEL_MAP, HORIZON_OPTIONS,
    load_and_aggregate, run_forecast, compute_summary
)

# =============================================================================
# Konfigurasi Halaman — HARUS PALING ATAS
# =============================================================================
st.set_page_config(
    page_title='Forecast Harga Pangan Italia',
    page_icon='🥩',
    layout='wide'
)

@st.cache_resource
def load_model_and_scaler(model_path: str, scaler_path: str):
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f'rgba({r}, {g}, {b}, {alpha})'

# =============================================================================
# CSS Tambahan
# =============================================================================
st.markdown("""
    <style>
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 8px;
            border-left: 4px solid #3498db;
        }
        .metric-label { font-size: 0.82rem; color: #666; margin-bottom: 2px; }
        .metric-value { font-size: 1.1rem; font-weight: 600; color: #222; }
        .alert-up   { border-left-color: #e74c3c !important; }
        .alert-down { border-left-color: #2ecc71 !important; }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# Header
# =============================================================================
st.title('🥩🥦🍎 Forecast Harga Pangan Italia')
st.markdown('Sistem prediksi harga rata-rata harian untuk komoditas **Buah (Fruit)**, **Sayuran (Vegetable)**, dan **Daging (Meat)**.')
st.divider()

# =============================================================================
# Sidebar — Input User
# =============================================================================
with st.sidebar:
    st.header('⚙️ Pengaturan')

    category = st.selectbox(
        'Pilih Kategori Komoditas',
        options=['Fruit', 'Vegetable', 'Meat'],
        format_func=lambda x: {'Fruit': '🍎 Fruit', 'Vegetable': '🥦 Vegetable', 'Meat': '🥩 Meat'}[x]
    )

    horizon_label = st.selectbox(
        'Horizon Forecast',
        options=list(HORIZON_OPTIONS.keys())
    )
    horizon = HORIZON_OPTIONS[horizon_label]

    show_hist_days = st.slider(
        'Tampilkan historis (hari terakhir)',
        min_value=30, max_value=365, value=90, step=30
    )

    run_btn = st.button('🚀 Jalankan Forecast', use_container_width=True, type='primary')

# =============================================================================
# Main — Jalankan Forecast
# =============================================================================
if run_btn:
    color = CATEGORY_COLORS[category]
    model_path, scaler_path = CATEGORY_MODEL_MAP[category]

    with st.spinner(f'Memuat model dan menjalankan forecast {category}...'):
        # Load model & scaler
        model  = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        # Load & agregasi data historis
        df_hist = load_and_aggregate('Variations_Food_Prices_Italian_Supermarkets.csv', category)

        # Jalankan forecast
        df_forecast = run_forecast(df_hist, model, scaler, horizon)
        summary     = compute_summary(df_forecast, df_hist)

    st.success(f'Forecast {category} — {horizon_label} berhasil dijalankan!')

    # -------------------------------------------------------------------------
    # Ringkasan Metrik
    # -------------------------------------------------------------------------
    st.subheader('📊 Ringkasan Forecast')

    change_val = float(summary['Perubahan (%)'].replace('%', '').replace('+', ''))
    alert_class = 'alert-up' if change_val > 5 else 'alert-down' if change_val < -5 else ''

    cols = st.columns(6)
    for col, (label, value) in zip(cols, summary.items()):
        extra_class = alert_class if label == 'Perubahan (%)' else ''
        col.markdown(f"""
            <div class="metric-card {extra_class}">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
        """, unsafe_allow_html=True)

    # Alert jika perubahan signifikan
    if change_val > 5:
        st.warning(f'⚠️ Harga **{category}** diprediksi **naik {change_val:.2f}%** dalam {horizon_label} ke depan. Pertimbangkan intervensi kebijakan.')
    elif change_val < -5:
        st.info(f'📉 Harga **{category}** diprediksi **turun {abs(change_val):.2f}%** dalam {horizon_label} ke depan.')

    st.divider()

    # -------------------------------------------------------------------------
    # Grafik Forecast
    # -------------------------------------------------------------------------
    st.subheader('📈 Grafik Forecast')

    df_hist_plot = df_hist.tail(show_hist_days)

    fig = go.Figure()

    # Historis
    fig.add_trace(go.Scatter(
        x=df_hist_plot['day'],
        y=df_hist_plot['avg_price'],
        mode='lines',
        name='Historis',
        line=dict(color='#95a5a6', width=1.5),
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=df_forecast['Tanggal'],
        y=df_forecast['Forecast'],
        mode='lines',
        name=f'Forecast ({horizon_label})',
        line=dict(color=color, width=2.5, dash='dash'),
        fill='tonexty',
        fillcolor=hex_to_rgba(color, 0.08)
    ))

    # Garis pemisah historis vs forecast
    fig.add_vline(
        x=df_hist['day'].max().timestamp() * 1000,
        line_dash='dot',
        line_color='black',
        annotation_text='Mulai Forecast',
        annotation_position='top right'
    )

    fig.update_layout(
        title=f'Harga Rata-rata Harian — {category}',
        xaxis_title='Tanggal',
        yaxis_title='Avg Price (€)',
        hovermode='x unified',
        legend=dict(orientation='h', y=1.02, x=0),
        height=420,
        yaxis=dict(range=[
            min(df_hist_plot['avg_price'].min(), df_forecast['Forecast'].min()) * 0.98,
            max(df_hist_plot['avg_price'].max(), df_forecast['Forecast'].max()) * 1.02
        ]),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # -------------------------------------------------------------------------
    # Tabel Forecast
    # -------------------------------------------------------------------------
    st.subheader('📋 Tabel Forecast')

    df_display = df_forecast.copy()
    df_display['Tanggal'] = df_display['Tanggal'].dt.strftime('%Y-%m-%d')
    df_display['Forecast (€)'] = df_display['Forecast'].apply(lambda x: f'€{x:.4f}')
    df_display = df_display.drop(columns='Forecast')

    st.dataframe(df_display, use_container_width=True, height=300)

    st.divider()

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------
    st.subheader('⬇️ Export Hasil')

    col_csv, col_excel = st.columns(2)

    # CSV
    csv_buffer = df_forecast.to_csv(index=False).encode('utf-8')
    col_csv.download_button(
        label='📄 Download CSV',
        data=csv_buffer,
        file_name=f'forecast_{category.lower()}_{horizon_label.replace(" ", "_")}.csv',
        mime='text/csv',
        use_container_width=True
    )

    # Excel
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df_forecast.to_excel(writer, index=False, sheet_name='Forecast')
        df_hist.tail(show_hist_days).rename(columns={'day': 'Tanggal', 'avg_price': 'Harga Aktual (€)'}).to_excel(
            writer, index=False, sheet_name='Historis'
        )
    col_excel.download_button(
        label='📊 Download Excel',
        data=excel_buffer.getvalue(),
        file_name=f'forecast_{category.lower()}_{horizon_label.replace(" ", "_")}.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        use_container_width=True
    )

else:
    # Tampilan awal sebelum forecast dijalankan
    st.info('👈 Pilih kategori dan horizon di sidebar, lalu klik **Jalankan Forecast**.')

    st.markdown("""
    ### Panduan Penggunaan
    1. **Pilih kategori** komoditas: Fruit, Vegetable, atau Meat
    2. **Pilih horizon** forecast: 30, 90, atau 180 hari ke depan
    3. **Atur tampilan historis** — berapa hari terakhir yang ingin ditampilkan di grafik
    4. Klik **Jalankan Forecast**
    5. Lihat grafik, tabel, dan **download hasilnya** dalam format CSV atau Excel

    ### Kriteria Interpretasi
    | Perubahan Harga | Status |
    |---|---|
    | Naik > 5% | ⚠️ Perlu perhatian — pertimbangkan intervensi |
    | -5% s.d. +5% | ✅ Stabil |
    | Turun > 5% | 📉 Harga cenderung turun |
    """)