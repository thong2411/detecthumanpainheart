import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
import os

# ===================== PATH =====================
DATA_BILSTM = "data"
DATA_LSTM = "data_1"

# ===================== LOAD MODELS =====================
@st.cache_resource
def load_models():
    model_lstm = tf.keras.models.load_model(
        "model2.h5", compile=False
    )
    model_bilstm = tf.keras.models.load_model(
        "modelnew.h5", compile=False
    )
    return model_lstm, model_bilstm

# ===================== LOAD DATA =====================
@st.cache_resource
def load_test_data():
    X_test = np.load(os.path.join(DATA_BILSTM, "X_test.npy"))
    Y_test = np.load(os.path.join(DATA_BILSTM, "Y_test.npy"))

    X_test_1 = np.load(os.path.join(DATA_LSTM, "X_test_1.npy"))
    Y_test_1 = np.load(os.path.join(DATA_LSTM, "Y_test_1.npy"))

    return X_test, Y_test, X_test_1, Y_test_1


model_lstm, model_bilstm = load_models()
X_test, Y_test, X_test_1, Y_test_1 = load_test_data()

# ===================== UI =====================
st.title("📊 So sánh LSTM vs Bi-LSTM")

st.write("""
- **LSTM:** test trên `data_1/X_test_1.npy`
- **Bi-LSTM:** test trên `data/X_test.npy`
""")

# ===================== SHAPE CHECK =====================
st.subheader("🔍 Kiểm tra dữ liệu")

st.write("LSTM X_test_1:", X_test_1.shape)
st.write("LSTM Y_test_1:", Y_test_1.shape)

st.write("Bi-LSTM X_test:", X_test.shape)
st.write("Bi-LSTM Y_test:", Y_test.shape)

if len(X_test) != len(Y_test) or len(X_test_1) != len(Y_test_1):
    st.error("❌ Số samples X và Y không khớp")
    st.stop()

# ===================== PREDICT =====================
with st.spinner("🔄 Đang dự đoán... (file lớn)"):
    y_pred_lstm = model_lstm.predict(X_test_1, batch_size=32).ravel()
    y_pred_bilstm = model_bilstm.predict(X_test, batch_size=32).ravel()

y_lstm_bin = (y_pred_lstm > 0.5).astype(int)
y_bilstm_bin = (y_pred_bilstm > 0.5).astype(int)

# ===================== METRICS =====================
def calc_metrics(y_true, y_prob, y_bin):
    return {
        "Accuracy": accuracy_score(y_true, y_bin),
        "Precision": precision_score(y_true, y_bin),
        "Recall": recall_score(y_true, y_bin),
        "F1-score": f1_score(y_true, y_bin),
        "ROC-AUC": roc_auc_score(y_true, y_prob)
    }

metrics_lstm = calc_metrics(Y_test_1, y_pred_lstm, y_lstm_bin)
metrics_bilstm = calc_metrics(Y_test, y_pred_bilstm, y_bilstm_bin)

# ===================== TABLE =====================
st.subheader("📈 Bảng so sánh hiệu năng")

compare_df = pd.DataFrame({
    "Metric": metrics_lstm.keys(),
    "LSTM (data_1)": metrics_lstm.values(),
    "Bi-LSTM (data)": metrics_bilstm.values()
})

st.dataframe(compare_df, use_container_width=True)

st.subheader("Biểu đồ trực quan")

fig, ax = plt.subplots(figsize=(9, 5))

x = np.arange(len(compare_df["Metric"]))
width = 0.35

bars_lstm = ax.bar(
    x - width/2,
    compare_df["LSTM (data_1)"],
    width,
    label="LSTM"
)

bars_bilstm = ax.bar(
    x + width/2,
    compare_df["Bi-LSTM (data)"],
    width,
    label="Bi-LSTM"
)

# 🔍 Zoom trục Y
ax.set_ylim(0.8, 1.0)

ax.set_xticks(x)
ax.set_xticklabels(compare_df["Metric"], rotation=30)
ax.set_ylabel("Score")
ax.set_title("So sánh hiệu năng LSTM và Bi-LSTM (zoom)")
ax.legend()

# 🔢 Hiển thị giá trị trên đầu mỗi cột
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.005,               # khoảng cách chữ
            f"{height:.3f}",              # làm tròn 3 chữ số
            ha="center",
            va="bottom",
            fontsize=9
        )

add_value_labels(bars_lstm)
add_value_labels(bars_bilstm)

st.pyplot(fig)

st.subheader("📈 Xu hướng hiệu năng tổng thể")

fig2, ax2 = plt.subplots(figsize=(8, 5))

ax2.plot(
    compare_df["Metric"],
    compare_df["LSTM (data_1)"],
    marker="o",
    label="LSTM"
)

ax2.plot(
    compare_df["Metric"],
    compare_df["Bi-LSTM (data)"],
    marker="s",
    label="Bi-LSTM"
)

ax2.set_ylim(0, 1)
ax2.set_ylabel("Score")
ax2.set_xlabel("Metric")
ax2.legend()
ax2.grid(True)

st.pyplot(fig2)

# ===================== CONFUSION MATRIX =====================
st.subheader("🔲 Confusion Matrix")

col1, col2 = st.columns(2)

with col1:
    st.write("LSTM")
    st.write(confusion_matrix(Y_test_1, y_lstm_bin))

with col2:
    st.write("Bi-LSTM")
    st.write(confusion_matrix(Y_test, y_bilstm_bin))

# ===================== ROC CURVE =====================
st.subheader("📉 ROC Curve")

fpr1, tpr1, _ = roc_curve(Y_test_1, y_pred_lstm)
fpr2, tpr2, _ = roc_curve(Y_test, y_pred_bilstm)

fig, ax = plt.subplots()
ax.plot(fpr1, tpr1, label="LSTM")
ax.plot(fpr2, tpr2, label="Bi-LSTM")
ax.plot([0, 1], [0, 1], linestyle="--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()

st.pyplot(fig)

