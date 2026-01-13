import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import base64

def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        /* Dark overlay for readability */
        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.45);
            z-index: -1;
        }}

        /* Make text readable */
        h1, h2, h3, p, label, span {{
            color: white !important;
        }}

        /* Sidebar styling (if you add one later) */
        [data-testid="stSidebar"] {{
            background-color: rgba(0,0,0,0.7);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("b-1024x768-1.jpg")


st.set_page_config(page_title="Outliers & Anomalies", layout="wide")
st.title("Outliers, Anomalies & K-Means Clustering")
st.caption("Analyze datasets, detect outliers, and visualize clusters with centroids.")

# -----------------------------
# 1. Dataset selection
# -----------------------------
dataset_choice = st.radio(
    "Choose Dataset",
    ["shadowpen_data_live.csv", "clean_data.csv"]
)

try:
    df = pd.read_csv(dataset_choice)
except FileNotFoundError:
    st.error(f"Dataset {dataset_choice} not found in this directory.")
    st.stop()

st.subheader(f"Dataset: {dataset_choice}")
st.dataframe(df.head())

# -----------------------------
# 2. Missing values analysis
# -----------------------------
st.subheader("Missing Values Summary")
missing = df.isna().sum().to_frame("missing_count")
missing["missing_percent"] = (missing["missing_count"] / len(df)) * 100
st.dataframe(missing.style.format({"missing_percent": "{:.1f}%"}))

# -----------------------------
# 3. Numeric columns selection
# -----------------------------
num_cols = df.select_dtypes(include=np.number).columns.tolist()
if len(num_cols) < 2:
    st.error("Need at least 2 numeric columns to analyze.")
    st.stop()

selected_cols = st.multiselect(
    "Select numeric columns for analysis",
    num_cols,
    default=num_cols[:2]
)
data = df[selected_cols].dropna()

# -----------------------------
# 4. Outlier detection (IQR)
# -----------------------------
st.subheader("Outlier Detection (IQR Method)")

def detect_outliers_iqr(dataframe):
    mask = pd.Series(False, index=dataframe.index)
    for col in dataframe.columns:
        Q1 = dataframe[col].quantile(0.25)
        Q3 = dataframe[col].quantile(0.75)
        IQR = Q3 - Q1
        mask |= (dataframe[col] < Q1 - 1.5*IQR) | (dataframe[col] > Q3 + 1.5*IQR)
    return mask

outliers = detect_outliers_iqr(data)
outlier_pct = outliers.mean() * 100
st.metric("Outliers detected", f"{outlier_pct:.1f}%")
clean_data = data[~outliers]

# -----------------------------
# 5. K-Means clustering
# -----------------------------
st.subheader("K-Means Clustering (Unsupervised)")
k = st.slider("Number of clusters (k)", 2, 6, 4)
scaler = StandardScaler()
scaled = scaler.fit_transform(clean_data)

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(scaled)

# Predict clusters for all cleaned data
all_labels = kmeans.predict(scaled)
centroids = kmeans.cluster_centers_

# -----------------------------
# 6. Visualization: Scatter plot with centroids
# -----------------------------
# -----------------------------
# 6. Visual Analysis
# -----------------------------
st.subheader("Visual Analysis")

col1, col2 = st.columns(2)

# -------- BOX PLOT --------
with col1:
    st.markdown("### Boxplot (Outliers Visible)")
    fig_box, ax_box = plt.subplots(figsize=(6, 4))
    ax_box.boxplot(clean_data.values, labels=clean_data.columns)
    ax_box.set_title("Boxplot After Cleaning")
    st.pyplot(fig_box)

# -------- RAW VS CLEANED --------
with col2:
    st.markdown("### Raw vs Cleaned Points")
    fig_raw, ax_raw = plt.subplots(figsize=(6, 4))
    ax_raw.scatter(
        data.iloc[:, 0],
        data.iloc[:, 1],
        alpha=0.4,
        label="Raw Data"
    )
    ax_raw.scatter(
        clean_data.iloc[:, 0],
        clean_data.iloc[:, 1],
        alpha=0.8,
        label="Cleaned Data"
    )
    ax_raw.set_xlabel(clean_data.columns[0])
    ax_raw.set_ylabel(clean_data.columns[1])
    ax_raw.legend()
    st.pyplot(fig_raw)

# -----------------------------
# 7. K-Means Clustering Plot
# -----------------------------
st.subheader("K-Means Clusters with Centroids")

# Inverse transform centroids back to original scale
centroids_original = scaler.inverse_transform(centroids)

fig, ax = plt.subplots(figsize=(8, 6))

# Scatter clustered points
ax.scatter(
    clean_data.iloc[:, 0],
    clean_data.iloc[:, 1],
    c=all_labels,
    cmap="tab10",
    alpha=0.7,
    s=60
)

# Draw BIG centroid rings
for i, centroid in enumerate(centroids_original):
    ax.scatter(
        centroid[0],
        centroid[1],
        s=500,                 # BIG ring
        facecolors='none',
        edgecolors='black',
        linewidths=3
    )
    ax.text(
        centroid[0],
        centroid[1],
        f"C{i+1}",
        fontsize=12,
        weight="bold",
        ha="center",
        va="center"
    )

ax.set_xlabel(clean_data.columns[0])
ax.set_ylabel(clean_data.columns[1])
ax.set_title("Cleaned Data with K-Means Centroids")

st.pyplot(fig)


# -----------------------------
# 7. Explanation
# -----------------------------
with st.expander("How outliers and clusters are calculated"):
    st.markdown("""
**Outlier Detection (IQR Method)**
- Points outside 1.5×IQR of each numeric column are marked as outliers.

**K-Means Clustering**
- Unsupervised algorithm trained on all cleaned points.
- Cluster labels assigned to all points.
- Centroids highlighted with black rings in the scatter plot.

**Dual Dataset**
- Choose between live data or backup.
- Visualizations update automatically.
""")
