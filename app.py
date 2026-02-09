import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import math
import pandas as pd
import time

# ==========================================
# Streamlit 页面配置
# ==========================================
st.set_page_config(page_title="Blast PPV Simulator", layout="wide")
st.title("💥 Blast Induced Ground Vibration Simulator")
st.markdown("---")

# ==========================================
# 1. 参数设置 (UI Sidebar)
# ==========================================
st.sidebar.header("1. Site Parameters")
b = st.sidebar.number_input("Constant b", value=3000.0)
n = st.sidebar.number_input("Constant n", value=3.0)
K = st.sidebar.number_input("Constant K", value=700.0)
A = st.sidebar.number_input("Attenuation A", value=0.7)
B = st.sidebar.number_input("Attenuation B", value=1.5)
Cp = st.sidebar.number_input("P-wave Velocity (Cp)", value=4844.0)
Cs = st.sidebar.number_input("S-wave Velocity (Cs)", value=2800.0)

st.sidebar.header("2. Hole Parameters")
charge_length = st.sidebar.number_input("Charge Length (m)", value=15.0)
stemming = st.sidebar.number_input("Stemming (m)", value=5.0)
borehole_diameter = st.sidebar.number_input("Borehole Diameter (m)", value=0.25)
density = st.sidebar.number_input("Density (kg/m³)", value=1600.0)
vod = st.sidebar.number_input("VOD (m/s)", value=4000.0)

# --- 修正：Detonation Point 为数值输入 ---
det_point = st.sidebar.number_input(
    "Detonation Point (m from bottom)", 
    min_value=0.0, 
    max_value=charge_length, 
    value=0.0, 
    step=0.5,
    help="Distance from the bottom of the charge (0 = Bottom initiation)."
)

# 计算依赖参数
Hole_length = charge_length + stemming

st.sidebar.header("3. Simulation Settings")
M = st.sidebar.slider("Discretization Segments (M)", 10, 100, 30)
grid_step = st.sidebar.number_input("Grid Step (m)", value=0.5)

# ==========================================
# 主程序逻辑 (点击按钮后执行)
# ==========================================
def run_simulation():
    # 记录开始时间
    start_time = time.time()
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    status_text.text("Initializing parameters...")

    # --- Derived Parameters ---
    l = charge_length / M
    we = math.pi * density * (borehole_diameter / 2) ** 2 * l

    # --- Time parameters ---
    t_det_up = (charge_length - det_point) / vod
    t_det_low = det_point / vod
    t_cal = 5 * max(t_det_up, t_det_low)
    dt = max(t_det_up, t_det_low) / M

    # --- Receiver grid ---
    # 使用用户输入的 grid_step
    xvalues = np.arange(0, 20.01, grid_step)
    yvalues = np.arange(-40, 0.01, grid_step) 
    tvalues = np.arange(0, t_cal, dt)

    # ==========================================
    # 2. 构建震源 (Build Sources with Mirrors)
    # ==========================================
    sources = []
    for m in range(M):
        z_m = m * l                    
        z = z_m - Hole_length         
        t_det_m = abs(z_m - det_point) / vod
        sources.append((z, t_det_m, we, False))    
        sources.append((-z, t_det_m, we, True))    

    # ==========================================
    # 3. 计算核心函数 (Calculation Function) - 完全保留原始逻辑
    # ==========================================
    def source_contribution_at_time(x_rec, y_rec, z_src, t, t_det_m):
        if t < t_det_m:
            return 0.0, 0.0

        tm = b * (t - t_det_m)
        Rm = math.hypot(x_rec, (y_rec - z_src)) 
        if Rm <= 1e-6:
            return 0.0, 0.0

        tpm = tm - b * (Rm / Cp)
        tsm = tm - b * (Rm / Cs)

        sintheta = x_rec / Rm
        costheta = -(y_rec - z_src) / Rm

        gama_n = 0.991 if n == 3 else 0.0455

        VRm = 0.0
        VZm = 0.0

        if tpm > 0:
            VRm = (gama_n * K * (we ** A) / (Rm ** B) *
                   (tpm ** n - 2 * n * tpm ** (n - 1) + n * (n - 1) * tpm ** (n - 2)) *
                   (1 - 2 * (Cs ** 2 / Cp ** 2) * costheta ** 2) * math.exp(-tpm))

        if tsm > 0:
            VZm = (gama_n * K * (we ** A) / (Rm ** B) *
                   (tsm ** n - 2 * n * tsm ** (n - 1) + n * (n - 1) * tsm ** (n - 2)) *
                   (Cp / Cs * 2 * sintheta * costheta) * math.exp(-tsm))

        UR = VRm * sintheta + VZm * costheta
        UZ = -VRm * costheta + VZm * sintheta

        return UR, UZ

    # ==========================================
    # 4. 时程叠加计算 (Time-History Loop)
    # ==========================================
    status_text.text("Calculating Time-History... This may take a moment.")
    PPV_final = []  
    
    total_points = len(xvalues) * len(yvalues)
    
    # 简单的进度计算器
    for xi, x in enumerate(xvalues):
        # Update progress bar
        progress_perc = int((xi / len(xvalues)) * 100)
        progress_bar.progress(progress_perc)

        for yi, y in enumerate(yvalues):
            UR_t = np.zeros_like(tvalues)
            UZ_t = np.zeros_like(tvalues)

            for ti, t in enumerate(tvalues):
                UR_sum = 0.0
                UZ_sum = 0.0

                for (z_src, t_det_m, we_seg, is_mirror) in sources:
                    UR_c, UZ_c = source_contribution_at_time(x, y, z_src, t, t_det_m)
                    if is_mirror:
                        UR_sum += UR_c        
                        UZ_sum += -UZ_c       
                    else:
                        UR_sum += UR_c
                        UZ_sum += UZ_c

                UR_t[ti] = UR_sum
                UZ_t[ti] = UZ_sum

            vel_tot = np.sqrt(UR_t ** 2 + UZ_t ** 2)
            ppv_max = np.max(vel_tot)
            PPV_final.append([x, y, ppv_max])

    progress_bar.progress(100)
    status_text.text("Calculation complete. Generating Plot...")

    # 镜像到负X轴
    PPV_all = PPV_final + [[-x, y, v] for x, y, v in PPV_final if x != 0]

    # ==========================================
    # 6. 绘图 (Plotting) - 保留 Colorbar
    # ==========================================
    x_array = np.array([row[0] for row in PPV_all])
    y_array = np.array([row[1] for row in PPV_all])
    ppv_array = np.array([row[2] for row in PPV_all])

    x_unique = np.unique(x_array)
    y_unique = np.unique(y_array)
    x_grid, y_grid = np.meshgrid(x_unique, y_unique, indexing='ij')
    ppv_grid = np.zeros_like(x_grid)

    for x, y, v in PPV_all:
        ix = np.where(x_unique == x)[0][0]
        iy = np.where(y_unique == y)[0][0]
        ppv_grid[ix, iy] = v

    # 设置画布
    fig_size_base = 8
    fig, ax = plt.subplots(figsize=(fig_size_base, fig_size_base))

    levels = np.arange(0, 10001, 1000)

    # --- 核心绘图 ---
    contour = ax.contourf(x_grid, y_grid, np.clip(ppv_grid, 0, 10000), levels=levels, cmap='jet', antialiased=True)
    
    # --- 添加 Colorbar ---
    cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('PPV (mm/s)', rotation=270, labelpad=15)

    # -----------------------------------------------------------
    # --- 绘制 Borehole Rectangle (有边框版) ---
    # -----------------------------------------------------------
    # 1. 装药段 (Charge)
    rect_charge = patches.Rectangle(
        (-0.2, -Hole_length),
        width=0.4,
        height=charge_length,
        facecolor='gold',
        edgecolor='black',
        linewidth=1.2,
        zorder=10,
        label='Charge'
    )
    ax.add_patch(rect_charge)

    # 2. 填塞段 (Stemming)
    rect_stemming = patches.Rectangle(
        (-0.2, -stemming),
        width=0.4,
        height=stemming,
        facecolor='darkblue',
        edgecolor='black',
        linewidth=1.2,
        zorder=10,
        label='Stemming'
    )
    ax.add_patch(rect_stemming)

    # -----------------------------------------------------------

    # 移除坐标轴和白边
    ax.axis('off')
    ax.margins(0,0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.set_aspect('equal')

    # 显示结果
    st.pyplot(fig)
    
    elapsed = time.time() - start_time
    st.success(f"Simulation finished in {elapsed:.2f} seconds")

    # CSV Download
    ppv_data_export = pd.DataFrame(PPV_all, columns=['x', 'y', 'ppv'])
    csv = ppv_data_export.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV Data", csv, "ppv_data.csv", "text/csv")

# ==========================================
# 主程序入口
# ==========================================
if st.button("Run Simulation", type="primary"):
    run_simulation()
else:
    st.info("Adjust parameters on the left and click 'Run Simulation'.")