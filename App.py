import streamlit as st
import kagglehub
import pickle
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
st.set_page_config(page_title="Iris Dasboard App", layout="centered")
st.sidebar.header("Dashboard")
st.title("Selamat Datang di Aplikasi Proyek Data Mining")
st.write("Ini adalah aplikasi Streamlit pertamaku")
# Download latest version
path = kagglehub.dataset_download("himanshunakrani/iris-dataset")
print("Path to dataset files:", path)
#Tampilkan dataframe
st.subheader("Dataset iris")
st.dataframe(df)
st.write(df.columns.tolist())
#df[target] = data target
#df['label'] = df['variety'].map({0:'iris-setosa',1:'iris-versicolor',2:'iris-virganica})
class_counts = df['variety'].value_counts()
#distribusi kelas
st.subheader("distribusi jumlah data berdasarkan kelas")
fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(x=class_counts.index,y=class_counts.values,palette=["red","green","yellow"],ax=ax)
ax.set_ylabel("jumlah data")
ax.set_xlabel("varietas")
ax.set_title("distribusi kelas iris")
st.pyplot(fig)
#korelasi fitur
st.subheader("Korelasi antar fitur dalam dataset")
#Input interaktif

