import numpy as np
import matplotlib.pyplot as plt
import math
import time
import streamlit as st
from matplotlib import patches

# ==========================================================
# Streamlit 页面设置
# ==========================================================
st.set_page_config(layout="wide")
st.title("Blast-Induced PPV Cloud Tool")
st.caption("Heelan–Blair based single-hole PPV simulation")

# ==========================================================
# 侧边栏参数输入
# ==========================================================
st.sidebar.header("Blast Parameters")

charge_length = st.sidebar.slider("Charge length (m)", 5.0, 30.0, 15.0)
stemming = st.sidebar.slider("Stemming (m)", 1.0, 10.0, 5.0)
density = st.sidebar.number_input("Explosive density (kg/m³)", 1200, 2000, 1600)
vod = st.sidebar.number_input("VOD (m/s)", 2000, 6000, 4000)
M = st.sidebar.slider("Charge discretization (M)", 10, 60, 30)

run_button = st.sidebar.button("Run Simulation")

# ==========================================================
# 核心计算函数（几乎原封不动）
# ==========================================================
def run_ppv_model():
    start_time = time.time()

    # --- Constants ---
    b = 3000
    n = 3
    K = 700
    A = 0.7
    B = 1.5
    Cp = 4844
    Cs = 2800

    Hole_length = charge_length + stemming
    borehole_diameter = 0.25

    l = charge_length / M
    we = math.pi * density * (borehole_diameter / 2) ** 2 * l
    det_point = 0

    t_det_up = (charge_length - det_point) / vod
    t_cal = 5 * t_det_up
    dt = t_det_up / M

    xvalues = np.arange(0, 20.01, 0.5)
    yvalues = np.arange(-40, 0.01, 0.5)
    tvalues = np.arange(0, t_cal, dt)

    # --- Build sources ---
    sources = []
    for m in range(M):
        z_m = m * l
        z = z_m - Hole_length
        t_det_m = abs(z_m - det_point) / vod
        sources.append((z, t_det_m, False))
        sources.append((-z, t_det_m, True))

    def source_contribution(x_rec, y_rec, z_src, t, t_det_m):
        if t < t_det_m:
            return 0.0, 0.0

        tm = b * (t - t_det_m)
        Rm = math.hypot(x_rec, (y_rec - z_src))
        if Rm < 1e-6:
            return 0.0, 0.0

        tpm = tm - b * (Rm / Cp)
        tsm = tm - b * (Rm / Cs)

        sintheta = x_rec / Rm
        costheta = -(y_rec - z_src) / Rm
        gama_n = 0.991

        VRm = 0.0
        VZm = 0.0

        if tpm > 0:
            VRm = (gama_n * K * (we ** A) / (Rm ** B) *
                   (tpm ** n - 2 * n * tpm ** (n - 1) + n * (n - 1) * tpm ** (n - 2)) *
                   (1 - 2 * (Cs ** 2 / Cp ** 2) * costheta ** 2) *
                   math.exp(-tpm))

        if tsm > 0:
            VZm = (gama_n * K * (we ** A) / (Rm ** B) *
                   (tsm ** n - 2 * n * tsm ** (n - 1) + n * (n - 1) * tsm ** (n - 2)) *
                   (Cp / Cs * 2 * sintheta * costheta) *
                   math.exp(-tsm))

        UR = VRm * sintheta + VZm * costheta
        UZ = -VRm * costheta + VZm * sintheta
        return UR, UZ

    PPV = []

    for x in xvalues:
        for y in yvalues:
            UR_t = np.zeros_like(tvalues)
            UZ_t = np.zeros_like(tvalues)

            for ti, t in enumerate(tvalues):
                UR_sum = 0.0
                UZ_sum = 0.0
                for z_src, t_det_m, is_mirror in sources:
                    ur, uz = source_contribution(x, y, z_src, t, t_det_m)
                    UR_sum += ur
                    UZ_sum += -uz if is_mirror else uz
                UR_t[ti] = UR_sum
                UZ_t[ti] = UZ_sum

            vel = np.sqrt(UR_t ** 2 + UZ_t ** 2)
            PPV.append([x, y, np.max(vel)])

    PPV = np.array(PPV)

    idx = np.argmax(PPV[:, 2])
    max_ppv = PPV[idx]

    return PPV, max_ppv, time.time() - start_time

# ==========================================================
# 运行并绘图
# ==========================================================
if run_button:
    with st.spinner("Running PPV simulation..."):
        PPV, max_ppv, runtime = run_ppv_model()

    st.success(f"Completed in {runtime:.2f} s")

    st.metric("Maximum PPV (mm/s)", f"{max_ppv[2]:.2f}")
    st.write(f"Location: x = {max_ppv[0]:.2f} m, y = {max_ppv[1]:.2f} m")

    # --- Grid ---
    x_unique = np.unique(PPV[:, 0])
    y_unique = np.unique(PPV[:, 1])
    X, Y = np.meshgrid(x_unique, y_unique, indexing="ij")
    Z = PPV[:, 2].reshape(len(x_unique), len(y_unique))

    fig, ax = plt.subplots(figsize=(7, 7))
    levels = np.arange(0, 10001, 1000)
    ax.contourf(X, Y, Z, levels=levels, cmap="jet")
    ax.set_aspect("equal")
    ax.axis("off")

    # Borehole
    ax.add_patch(patches.Rectangle((-0.2, -charge_length-stemming),
                                   0.4, charge_length,
                                   facecolor="gold", edgecolor="black"))
    ax.add_patch(patches.Rectangle((-0.2, -stemming),
                                   0.4, stemming,
                                   facecolor="darkblue", edgecolor="black"))

    st.pyplot(fig)
