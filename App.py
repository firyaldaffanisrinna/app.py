import streamlit as st
import pandas as pd
import pickle
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Konfigurasi halaman
st.set_page_config(page_title="Iris Dashboard App", layout="centered")
st.sidebar.header("Dashboard")

# Judul aplikasi
st.title("Selamat Datang di Aplikasi Proyek Data Mining")
st.write("Ini adalah aplikasi Streamlit pertamaku")

# Load dataset
df = pd.read_csv("model/iris.csv")

# Tambahkan kolom 'variety' berdasarkan label target
df['variety'] = df['target'].map({0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'})

# Tampilkan dataset
st.subheader("Dataset Iris")
st.dataframe(df)
st.write("Kolom-kolom:", df.columns.tolist())

# Distribusi kelas
st.subheader("Distribusi Jumlah Data Berdasarkan Kelas")
class_counts = df['variety'].value_counts()
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(x=class_counts.index, y=class_counts.values, palette=["red", "green", "yellow"], ax=ax)
ax.set_ylabel("Jumlah Data")
ax.set_xlabel("Varietas")
ax.set_title("Distribusi Kelas Iris")
st.pyplot(fig)

# Korelasi antar fitur
st.subheader("Korelasi Antar Fitur dalam Dataset")
fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
sns.heatmap(df.drop(columns=['variety']).corr(), annot=True, cmap="YlGnBu", ax=ax_corr)
st.pyplot(fig_corr)
