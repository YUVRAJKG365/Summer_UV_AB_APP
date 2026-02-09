import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib import cm
import plotly.graph_objects as go
import plotly.express as px
import json
from mpl_toolkits.mplot3d import Axes3D
import io

# -----------------------------------
# Page config
# -----------------------------------
st.set_page_config(
    page_title="UV Radiation Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4, #FFEAA7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        text-align: center;
        padding: 20px;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        border-left: 5px solid #4ECDC4;
        padding-left: 15px;
        margin-top: 30px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .stButton>button {
        background: linear-gradient(45deg, #FF6B6B, #FFE66D);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(255,107,107,0.4);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# Paths
# -----------------------------------
MODEL_PATH = r"C:\Users\yuvra\PycharmProjects\Summer_UV_AB_APP\radiation_prediction_model.pkl"
STATS_PATH = r"C:\Users\yuvra\PycharmProjects\Summer_UV_AB_APP\train_stats.json"

# -----------------------------------
# Load model & stats
# -----------------------------------
try:
    model = joblib.load(MODEL_PATH)
    with open(STATS_PATH, "r") as f:
        train_stats = json.load(f)
    st.sidebar.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# -----------------------------------
# Manual standardization function
# -----------------------------------
def manual_standardize(value, mean, std):
    return (value - mean) / std


# -----------------------------------
# Prediction function using the model
# -----------------------------------
def predict_uv_intensity(time_value, altitude_value, uv_type_value):
    """Predict UV intensity for given parameters using the loaded model"""
    # Prepare single data point
    data_point = pd.DataFrame({
        "Time_Numeric": [time_value],
        "Altitude": [altitude_value],
        "UV_Type_UVA": [1 if uv_type_value == "UV-A" else 0],
        "UV_Type_UVB": [0 if uv_type_value == "UV-A" else 1]
    })

    # Standardize using training stats
    data_point["Time_Numeric"] = manual_standardize(
        data_point["Time_Numeric"].iloc[0],
        train_stats["Time_Numeric"]["mean"],
        train_stats["Time_Numeric"]["std"]
    )
    data_point["Altitude"] = manual_standardize(
        data_point["Altitude"].iloc[0],
        train_stats["Altitude"]["mean"],
        train_stats["Altitude"]["std"]
    )

    # Predict using the model
    prediction = model.predict(data_point[["Time_Numeric", "Altitude", "UV_Type_UVA", "UV_Type_UVB"]])
    return prediction[0]


# -----------------------------------
# Generate 3D surface data using model predictions
# -----------------------------------
def generate_3d_surface_data(time_range, altitude_range, uv_type):
    """Generate 3D surface data using model predictions"""
    X, Y = np.meshgrid(time_range, altitude_range)
    Z = np.zeros_like(X)

    # Predict for each point in the grid
    for i in range(len(altitude_range)):
        for j in range(len(time_range)):
            Z[i, j] = predict_uv_intensity(time_range[j], altitude_range[i], uv_type)

    return X, Y, Z


# -----------------------------------
# Header
# -----------------------------------
st.markdown('<h1 class="main-header">🌈 UV Radiation Prediction Dashboard</h1>', unsafe_allow_html=True)

# -----------------------------------
# Sidebar Configuration
# -----------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration Panel")

    # Theme Selection
    graph_theme = st.selectbox(
        "🎨 Select Graph Theme",
        ["Rainbow", "Viridis", "Plasma", "Inferno", "Magma", "Sunset", "Ocean"]
    )

    # 3D Plot Resolution
    resolution = st.slider("3D Plot Resolution", 10, 50, 20,
                           help="Higher values = smoother surface but slower")

    # Color Customization
    st.markdown("### 🎨 Customize Colors")
    line_color = st.color_picker("Line Color", "#FF6B6B")
    fill_color = st.color_picker("Fill Color", "#4ECDC4")

    st.markdown("---")
    st.markdown("## 📊 Quick Stats")
    st.markdown("*All graphs use actual model predictions*")

# -----------------------------------
# Input Section
# -----------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    
    location_name = st.text_input("📍 Location Name", "Unseen Location")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    
    altitude = st.slider(
        "🏔 Altitude (meters)",
        min_value=0,
        max_value=6000,
        value=1200,
        step=50,
        help="Adjust altitude to see its effect on UV radiation"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    
    uv_type = st.selectbox(
        "☀ Radiation Type",
        ["UV-A", "UV-B"],
        help="Select between UV-A (315-400nm) and UV-B (280-315nm) radiation"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Time range input
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    start_time = st.slider("🌅 Start Time", 0, 23, 6)
with col2:
    end_time = st.slider("🌇 End Time", 0, 23, 18)

time_numeric = np.arange(start_time, end_time + 1)

# -----------------------------------
# Generate Predictions for Main Plot
# -----------------------------------
predictions = []
for time in time_numeric:
    pred = predict_uv_intensity(time, altitude, uv_type)
    predictions.append(pred)

predictions = np.array(predictions)
baseline_uv = predictions * 0.92

# -----------------------------------
# Visualization Section
# -----------------------------------
st.markdown('<h2 class="sub-header">📊 All Graphs Generated by Model Predictions</h2>', unsafe_allow_html=True)

# Get color map
color_maps = {
    "Rainbow": cm.rainbow,
    "Viridis": cm.viridis,
    "Plasma": cm.plasma,
    "Inferno": cm.inferno,
    "Magma": cm.magma,
    "Sunset": cm.summer,
    "Ocean": cm.ocean
}
selected_cmap = color_maps[graph_theme]

# Create tabs for different visualizations
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Time Series Plot",
    "🌈 3D Surface Plot",
    "🌐 Contour & Heatmap",
    "📈 Comparative Analysis",
    "📋 Data Explorer"
])

with tab1:
    # Time Series Plot with Model Predictions
    st.markdown("### 📈 Time Series Prediction")

    fig1, ax1 = plt.subplots(figsize=(14, 6))

    # Create gradient fill under the line
    ax1.fill_between(time_numeric, 0, predictions, alpha=0.3,
                     color=fill_color, label="UV Intensity Area")

    # Plot predictions
    colors = [selected_cmap(i) for i in np.linspace(0, 1, len(time_numeric))]
    for i in range(len(time_numeric) - 1):
        ax1.plot(time_numeric[i:i + 2], predictions[i:i + 2],
                 color=colors[i], linewidth=3, marker='o')

    # Plot baseline
    ax1.bar(time_numeric, baseline_uv, alpha=0.6,
            color=[selected_cmap(i * 0.7) for i in np.linspace(0, 1, len(time_numeric))],
            label="Baseline UV")

    ax1.set_xlabel("Time (Hours)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("UV Radiation Intensity", fontsize=12, fontweight='bold')
    ax1.set_title(f"Model Predictions: {uv_type} at {altitude}m",
                  fontsize=14, fontweight='bold', pad=20)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')

    # Add prediction values as text
    for i, (x, y) in enumerate(zip(time_numeric, predictions)):
        ax1.text(x, y * 1.02, f'{y:.2f}', ha='center', va='bottom',
                 fontsize=8, fontweight='bold', color=colors[i])

    st.pyplot(fig1)

    # Download Time Series PNG
    buf = io.BytesIO()
    fig1.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    st.download_button(
        label="📥 Download Time Series (PNG)",
        data=buf.getvalue(),
        file_name=f"time_series_{location_name}_{uv_type}_{altitude}m.png",
        mime="image/png"
    )

    # Show prediction details
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("⏰ Current Time", f"{time_numeric[len(time_numeric) // 2]}:00")
    with col2:
        st.metric("🔺 Predicted UV", f"{predictions[len(predictions) // 2]:.4f}")
    with col3:
        st.metric("📊 Prediction Count", len(predictions))

with tab2:
    # 3D Surface Plot using Model Predictions
    st.markdown("### 🌐 3D Surface Plot (Model Generated)")

    # Generate 3D data using model
    alt_range = np.linspace(max(0, altitude - 1000), altitude + 1000, resolution)
    time_range = np.linspace(start_time, end_time, resolution)

    with st.spinner("🔄 Generating 3D surface using model predictions..."):
        X_grid, Y_grid, Z_grid = generate_3d_surface_data(time_range, alt_range, uv_type)

    # Interactive 3D Plot with Plotly
    fig_3d = go.Figure(data=[go.Surface(
        z=Z_grid,
        x=X_grid,
        y=Y_grid,
        colorscale=graph_theme.lower(),
        opacity=0.9,
        contours={
            "z": {"show": True, "usecolormap": True, "highlightcolor": "limegreen"}
        }
    )])

    # Add the current altitude prediction line
    current_predictions = []
    for time in time_range:
        pred = predict_uv_intensity(time, altitude, uv_type)
        current_predictions.append(pred)

    fig_3d.add_trace(go.Scatter3d(
        x=time_range,
        y=[altitude] * len(time_range),
        z=current_predictions,
        mode='lines+markers',
        line=dict(color='red', width=4),
        marker=dict(size=5, color='white'),
        name=f'Current Altitude ({altitude}m)'
    ))

    fig_3d.update_layout(
        title=f'3D UV Surface: Model Predictions for {uv_type}',
        scene=dict(
            xaxis_title='Time (Hours)',
            yaxis_title='Altitude (m)',
            zaxis_title='UV Intensity',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=800,
        height=600,
        autosize=True
    )

    st.plotly_chart(fig_3d, use_container_width=True)

    # Download Plotly 3D (PNG or HTML fallback)
    try:
        img_bytes = fig_3d.to_image(format="png")
        st.download_button(
            label="📥 Download 3D Surface (PNG)",
            data=img_bytes,
            file_name=f"3d_surface_{location_name}_{uv_type}_{altitude}m.png",
            mime="image/png"
        )
    except Exception:
        # Fallback: download as interactive HTML
        html = fig_3d.to_html(full_html=False, include_plotlyjs='cdn')
        st.download_button(
            label="📥 Download 3D Surface (HTML)",
            data=html,
            file_name=f"3d_surface_{location_name}_{uv_type}_{altitude}m.html",
            mime="text/html"
        )

    # Matplotlib 3D Plot
    st.markdown("### 🎨 Alternative 3D View")
    fig2 = plt.figure(figsize=(12, 8))
    ax2 = fig2.add_subplot(111, projection='3d')

    # Plot surface
    surf = ax2.plot_surface(X_grid, Y_grid, Z_grid, cmap=selected_cmap,
                            alpha=0.8, linewidth=0, antialiased=True)

    # Scatter plot for current predictions
    ax2.scatter(time_numeric, [altitude] * len(time_numeric), predictions,
                c='red', s=50, depthshade=True, label='Current Predictions')

    ax2.set_xlabel('Time (Hours)')
    ax2.set_ylabel('Altitude (m)')
    ax2.set_zlabel('UV Intensity')
    ax2.set_title(f'3D Model Predictions: {uv_type}')
    fig2.colorbar(surf, ax=ax2, shrink=0.5, aspect=5, label='UV Intensity')
    ax2.legend()

    st.pyplot(fig2)

    # Download Matplotlib 3D PNG
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png', bbox_inches='tight')
    buf2.seek(0)
    st.download_button(
        label="📥 Download 3D Matplotlib View (PNG)",
        data=buf2.getvalue(),
        file_name=f"3d_matplotlib_{location_name}_{uv_type}_{altitude}m.png",
        mime="image/png"
    )

with tab3:
    # Contour and Heatmap Plots from Model Predictions
    st.markdown("### 🗺 Contour & Heatmap (Model Generated)")

    # Generate data for contour plot
    contour_res = 30
    alt_contour = np.linspace(0, 6000, contour_res)
    time_contour = np.linspace(start_time, end_time, contour_res)

    with st.spinner("🔄 Generating contour data using model..."):
        X_contour, Y_contour, Z_contour = generate_3d_surface_data(time_contour, alt_contour, uv_type)

    col1, col2 = st.columns(2)

    with col1:
        # Contour Plot
        fig_contour, ax_contour = plt.subplots(figsize=(10, 6))
        contour = ax_contour.contourf(X_contour, Y_contour, Z_contour, 20,
                                      cmap=selected_cmap, alpha=0.8)

        # Highlight current altitude
        ax_contour.axhline(y=altitude, color='red', linestyle='--', linewidth=2,
                           label=f'Current Altitude: {altitude}m')

        # Plot current predictions
        ax_contour.scatter(time_numeric, [altitude] * len(time_numeric),
                           c=predictions, s=100, cmap=selected_cmap,
                           edgecolors='black', zorder=5)

        ax_contour.set_xlabel('Time (Hours)')
        ax_contour.set_ylabel('Altitude (m)')
        ax_contour.set_title(f'Contour Plot: Model Predictions\n{uv_type} Radiation')
        plt.colorbar(contour, ax=ax_contour, label='UV Intensity')
        ax_contour.legend()
        ax_contour.grid(True, alpha=0.3)
        st.pyplot(fig_contour)

        # Download Contour PNG
        buf3 = io.BytesIO()
        fig_contour.savefig(buf3, format='png', bbox_inches='tight')
        buf3.seek(0)
        st.download_button(
            label="📥 Download Contour (PNG)",
            data=buf3.getvalue(),
            file_name=f"contour_{location_name}_{uv_type}_{altitude}m.png",
            mime="image/png"
        )

    with col2:
        # Heatmap
        fig_heat, ax_heat = plt.subplots(figsize=(10, 6))

        # Transpose for better visualization
        heatmap_data = Z_contour

        im = ax_heat.imshow(heatmap_data, aspect='auto', cmap=selected_cmap,
                            extent=[time_contour.min(), time_contour.max(),
                                    alt_contour.min(), alt_contour.max()],
                            origin='lower')

        # Add current altitude line
        alt_norm = (altitude - alt_contour.min()) / (alt_contour.max() - alt_contour.min())
        ax_heat.axhline(y=altitude, color='white', linestyle='--', linewidth=2,
                        label=f'Altitude: {altitude}m')

        ax_heat.set_xlabel('Time (Hours)')
        ax_heat.set_ylabel('Altitude (m)')
        ax_heat.set_title(f'Heatmap: Model Predictions\n{uv_type} Radiation')
        plt.colorbar(im, ax=ax_heat, label='UV Intensity')
        ax_heat.legend()

        st.pyplot(fig_heat)

        # Download Heatmap PNG
        buf4 = io.BytesIO()
        fig_heat.savefig(buf4, format='png', bbox_inches='tight')
        buf4.seek(0)
        st.download_button(
            label="📥 Download Heatmap (PNG)",
            data=buf4.getvalue(),
            file_name=f"heatmap_{location_name}_{uv_type}_{altitude}m.png",
            mime="image/png"
        )

with tab4:
    # Comparative Analysis using Model Predictions
    st.markdown("### 📊 Comparative Analysis")

    # Compare UV-A vs UV-B predictions
    col1, col2 = st.columns(2)

    with col1:
        # Generate predictions for both UV types
        predictions_uva = []
        predictions_uvb = []

        for time in time_numeric:
            pred_uva = predict_uv_intensity(time, altitude, "UV-A")
            pred_uvb = predict_uv_intensity(time, altitude, "UV-B")
            predictions_uva.append(pred_uva)
            predictions_uvb.append(pred_uvb)

        # Comparative Line Plot
        fig_compare, ax_compare = plt.subplots(figsize=(10, 6))

        ax_compare.plot(time_numeric, predictions_uva, 'o-', linewidth=3,
                        label='UV-A Predictions', color='blue', alpha=0.8)
        ax_compare.plot(time_numeric, predictions_uvb, 's-', linewidth=3,
                        label='UV-B Predictions', color='red', alpha=0.8)

        # Fill between
        ax_compare.fill_between(time_numeric, predictions_uva, predictions_uvb,
                                alpha=0.2, color='purple', label='Difference')

        ax_compare.set_xlabel('Time (Hours)')
        ax_compare.set_ylabel('UV Intensity')
        ax_compare.set_title(f'UV-A vs UV-B Model Predictions\nAltitude: {altitude}m')
        ax_compare.legend()
        ax_compare.grid(True, alpha=0.3)
        ax_compare.set_facecolor('#f8f9fa')

        st.pyplot(fig_compare)

        # Download Comparative Line Plot PNG
        buf5 = io.BytesIO()
        fig_compare.savefig(buf5, format='png', bbox_inches='tight')
        buf5.seek(0)
        st.download_button(
            label="📥 Download Comparison (PNG)",
            data=buf5.getvalue(),
            file_name=f"comparison_{location_name}_{uv_type}_{altitude}m.png",
            mime="image/png"
        )

    with col2:
        # Altitude Sensitivity Analysis
        st.markdown("#### 📈 Altitude Sensitivity")

        alt_test = np.linspace(0, 6000, 10)
        time_test = 12  # Noon

        sensitivity_predictions = []
        for alt in alt_test:
            pred = predict_uv_intensity(time_test, alt, uv_type)
            sensitivity_predictions.append(pred)

        fig_sens, ax_sens = plt.subplots(figsize=(10, 6))
        ax_sens.plot(alt_test, sensitivity_predictions, 'o-', linewidth=3,
                     color=line_color, markersize=8)
        ax_sens.fill_between(alt_test, 0, sensitivity_predictions,
                             alpha=0.3, color=fill_color)

        # Highlight current altitude
        current_pred = predict_uv_intensity(time_test, altitude, uv_type)
        ax_sens.axvline(x=altitude, color='red', linestyle='--',
                        label=f'Current: {altitude}m')
        ax_sens.scatter([altitude], [current_pred], color='red', s=100,
                        edgecolors='black', zorder=5)

        ax_sens.set_xlabel('Altitude (m)')
        ax_sens.set_ylabel(f'UV Intensity at {time_test}:00')
        ax_sens.set_title(f'Altitude Sensitivity: {uv_type} at Noon')
        ax_sens.legend()
        ax_sens.grid(True, alpha=0.3)

        st.pyplot(fig_sens)

        # Download Sensitivity Plot PNG
        buf6 = io.BytesIO()
        fig_sens.savefig(buf6, format='png', bbox_inches='tight')
        buf6.seek(0)
        st.download_button(
            label="📥 Download Sensitivity (PNG)",
            data=buf6.getvalue(),
            file_name=f"sensitivity_{location_name}_{uv_type}_{altitude}m.png",
            mime="image/png"
        )

with tab5:
    # Data Explorer Tab
    st.markdown("### 📋 Model Prediction Data")

    # Create detailed dataframe
    detailed_data = []
    for i, time in enumerate(time_numeric):
        detailed_data.append({
            'Time (Hour)': time,
            'Altitude (m)': altitude,
            'UV Type': uv_type,
            'Model Prediction': predictions[i],
            'Baseline (92%)': baseline_uv[i],
            'Difference': predictions[i] - baseline_uv[i],
            'Time Standardized': manual_standardize(time,
                                                    train_stats["Time_Numeric"]["mean"],
                                                    train_stats["Time_Numeric"]["std"]),
            'Altitude Standardized': manual_standardize(altitude,
                                                        train_stats["Altitude"]["mean"],
                                                        train_stats["Altitude"]["std"])
        })

    df_detailed = pd.DataFrame(detailed_data)

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📈 Max Prediction", f"{predictions.max():.4f}")
    with col2:
        st.metric("📉 Min Prediction", f"{predictions.min():.4f}")
    with col3:
        st.metric("📊 Average", f"{predictions.mean():.4f}")
    with col4:
        st.metric("🎯 Current Prediction",
                  f"{predictions[len(predictions) // 2]:.4f}")

    # Display data table
    st.dataframe(df_detailed, use_container_width=True, height=300)

    # Model Information
    with st.expander("🔍 Model Information"):
        st.write("**Model Statistics:**")
        st.json(train_stats)

        st.write("**Feature Importance:**")
        st.write(f"- Time_Numeric (Standardized)")
        st.write(f"- Altitude (Standardized)")
        st.write(f"- UV Type (One-hot encoded)")

        st.write("**Prediction Range:**")
        st.write(f"- Time: {start_time}:00 to {end_time}:00")
        st.write(f"- Altitude: {altitude}m")
        st.write(f"- UV Type: {uv_type}")

    # Download predictions
    csv = df_detailed.to_csv(index=False)
    st.download_button(
        label="📥 Download Model Predictions",
        data=csv,
        file_name=f"model_predictions_{location_name}_{uv_type}_{altitude}m.csv",
        mime="text/csv"
    )

# -----------------------------------
# Footer
# -----------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 20px;'>
    <p>🌍 <b>ALL GRAPHS GENERATED USING MODEL PREDICTIONS</b> • Machine Learning Dashboard</p>
    <p style='font-size: 0.8em;'>Every data point in every graph is generated by the trained model</p>
</div>
""", unsafe_allow_html=True)