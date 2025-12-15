import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import joblib  # Library untuk menyimpan model .pkl
import io

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from textblob import TextBlob

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="App Analisis Sentimen TikTok", page_icon="🤖", layout="wide"
)

st.title("🤖 Analisis Sentimen: Engineering Boys")
st.markdown(
    "Aplikasi *All-in-One*: Upload Data -> Latih Model -> **Download .pkl** -> Prediksi."
)

# ==========================================
# FUNGSI-FUNGSI BANTUAN
# ==========================================


# 1. Cleaning Text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Hapus angka & simbol
    text = re.sub(r"\s+", " ", text).strip()
    return text


# 2. Auto Labeling (TextBlob)
def get_sentiment_textblob(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.0:
        return "Positif"
    elif analysis.sentiment.polarity < 0.0:
        return "Negatif"
    else:
        return "Netral"


# ==========================================
# SIDEBAR: UPLOAD DATA
# ==========================================
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload file CSV (tiktok comments)", type=["csv"]
)

if uploaded_file is None:
    st.info("👋 Silakan upload file CSV di menu sebelah kiri untuk memulai.")
    st.stop()

# ==========================================
# PROSES UTAMA
# ==========================================

# Load Data
df = pd.read_csv(uploaded_file)

# Validasi Kolom
if "text" not in df.columns:
    st.error("❌ File CSV harus memiliki kolom bernama 'text'!")
    st.stop()

with st.spinner("Sedang memproses data dan melatih model..."):
    # 1. Preprocessing
    df["cleaned_text"] = df["text"].apply(clean_text)

    # Hapus duplikat & kosong
    df_clean = df.drop_duplicates(subset=["cleaned_text"])
    df_clean = df_clean[df_clean["cleaned_text"] != ""]

    # 2. Labeling
    if "label" in df_clean.columns:
        st.success("✅ Menggunakan label yang sudah ada dari file!")
    else:
        st.info("ℹ️ Melakukan auto-labeling (fallback) dengan TextBlob...")
        df_clean["label"] = df_clean["cleaned_text"].apply(get_sentiment_textblob)

    # 3. Vektorisasi (TF-IDF)
    # Gunakan ngram_range=(1,2) untuk menangkap frasa kata (misal: "tidak suka")
    tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    X = tfidf.fit_transform(df_clean["cleaned_text"])
    y = df_clean["label"]

    # 4. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- OVERSAMPLING MANUAL (Agar seimbang) ---
    # Kita tidak pakai imblearn agar tidak perlu install lib baru
    from sklearn.utils import resample

    # Gabungkan X_train dan y_train untuk resampling
    train_data = pd.DataFrame(X_train.toarray(), columns=tfidf.get_feature_names_out())
    train_data["label"] = y_train.values

    # Pisahkan berdasarkan kelas
    df_netral = train_data[train_data["label"] == "Netral"]
    df_positif = train_data[train_data["label"] == "Positif"]
    df_negatif = train_data[train_data["label"] == "Negatif"]

    # Oversample minoritas (Positif & Negatif) agar sama dengan Netral
    n_samples = len(df_netral)
    
    # Cek jika kosong untuk menghindari error
    if not df_positif.empty and n_samples > 0:
        df_positif_os = resample(df_positif, replace=True, n_samples=n_samples, random_state=42)
    else:
        df_positif_os = df_positif

    if not df_negatif.empty and n_samples > 0:
        df_negatif_os = resample(df_negatif, replace=True, n_samples=n_samples, random_state=42)
    else:
        df_negatif_os = df_negatif

    # Gabungkan kembali
    df_train_balanced = pd.concat([df_netral, df_positif_os, df_negatif_os])

    # Pisahkan X dan y lagi
    X_train_balanced = df_train_balanced.drop("label", axis=1).values
    y_train_balanced = df_train_balanced["label"]

    # 5. Training Model (Logistic Regression)
    # Logistic Regression biasanya lebih bagus untuk data tidak seimbang dibanding Naive Bayes
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_balanced, y_train_balanced)

    # 6. Evaluasi Awal
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)

st.success(f"✅ Model berhasil dilatih! Akurasi: {accuracy:.2%}")

# ==========================================
# TABS UTAMA
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Data & Visualisasi",
        "📈 Evaluasi Model",
        "💾 Download Model (.pkl)",
        "🔮 Prediksi Live",
        "🔍 Cari Kata",
    ]
)

# --- TAB 1: VISUALISASI ---
with tab1:
    st.subheader("Distribusi Sentimen")
    col1, col2 = st.columns(2)

    with col1:
        # Bar Chart
        fig_count = plt.figure(figsize=(6, 4))
        sns.countplot(
            x="label",
            data=df_clean,
            palette="pastel",
            order=["Positif", "Netral", "Negatif"],
        )
        plt.title("Jumlah Komentar per Sentimen")
        st.pyplot(fig_count)

    with col2:
        # Pie Chart
        sentiment_counts = df_clean["label"].value_counts()
        fig_pie, ax = plt.subplots()
        ax.pie(
            sentiment_counts,
            labels=sentiment_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=sns.color_palette("pastel"),
        )
        ax.axis("equal")
        st.pyplot(fig_pie)

    st.subheader("Word Cloud")
    all_text = " ".join(df_clean["cleaned_text"])
    wordcloud = WordCloud(width=800, height=300, background_color="white").generate(
        all_text
    )
    fig_wc = plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(fig_wc)

# --- TAB 2: EVALUASI ---
with tab2:
    st.subheader("Performa Model")

    col_ev1, col_ev2 = st.columns(2)

    with col_ev1:
        st.write("**Classification Report:**")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"))

    with col_ev2:
        st.write("**Confusion Matrix:**")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=model.classes_,
            yticklabels=model.classes_,
        )
        plt.xlabel("Prediksi")
        plt.ylabel("Aktual")
        st.pyplot(fig_cm)

# --- TAB 3: DOWNLOAD MODEL (.pkl) ---
with tab3:
    st.header("Simpan Model Kamu")
    st.info(
        "Download kedua file ini agar bisa digunakan di tempat lain tanpa perlu training ulang."
    )

    # Siapkan buffer memori untuk file download
    # 1. Simpan Model
    buffer_model = io.BytesIO()
    joblib.dump(model, buffer_model)
    buffer_model.seek(0)

    # 2. Simpan Vectorizer (PENTING: Pasangan Model)
    buffer_tfidf = io.BytesIO()
    joblib.dump(tfidf, buffer_tfidf)
    buffer_tfidf.seek(0)

    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        st.download_button(
            label="⬇️ Download Model (.pkl)",
            data=buffer_model,
            file_name="sentiment_model_tiktok.pkl",
            mime="application/octet-stream",
        )
        st.caption("File ini berisi otak/logika model.")

    with col_dl2:
        st.download_button(
            label="⬇️ Download Vectorizer (.pkl)",
            data=buffer_tfidf,
            file_name="tfidf_vectorizer.pkl",
            mime="application/octet-stream",
        )
        st.caption("File ini berisi otak/logika model.")

# --- TAB 4: PREDIKSI LIVE ---
with tab4:
    st.header("Coba Prediksi Sentimen")
    text_input = st.text_area("Masukkan komentar TikTok di sini:", "")

    if st.button("Prediksi"):
        if text_input.strip() == "":
            st.warning("⚠️ Masukkan teks dulu dong!")
        else:
            # 1. Clean
            clean_input = clean_text(text_input)
            # 2. Vectorize
            vec_input = tfidf.transform([clean_input])
            # 3. Predict
            pred = model.predict(vec_input)[0]

            # Tampilkan Hasil
            if pred == "Positif":
                st.success(f"**Hasil Prediksi: {pred}** 😄")
            elif pred == "Negatif":
                st.error(f"**Hasil Prediksi: {pred}** 😡")
            else:
                st.info(f"**Hasil Prediksi: {pred}** 😐")

# --- TAB 5: CARI KATA ---
with tab5:
    st.header("Caril Analisis Kata Spesifik")
    search_keyword = st.text_input("Ketik kata yang ingin dicari (misal: 'keren', 'mahal'):")

    if search_keyword:
        # Filter dataset
        filtered_df = df_clean[df_clean["cleaned_text"].str.contains(search_keyword.lower(), na=False)]
        
        if not filtered_df.empty:
            count = len(filtered_df)
            st.info(f"Ditemukan **{count}** komentar mengandung kata '**{search_keyword}**'.")
            
            col_search1, col_search2 = st.columns(2)
            
            with col_search1:
                st.write("**Distribusi Sentimen untuk kata ini:**")
                search_counts = filtered_df["label"].value_counts()
                
                # Bar Chart Khusus Kata Ini
                fig_search = plt.figure(figsize=(5, 3))
                sns.barplot(x=search_counts.index, y=search_counts.values, palette="viridis")
                plt.title(f"Sentimen untuk '{search_keyword}'")
                plt.xlabel("Sentimen")
                plt.ylabel("Jumlah")
                st.pyplot(fig_search)
                
            with col_search2:
                st.write("**Contoh Komentar:**")
                # Tampilkan beberapa sampel
                sample_comments = filtered_df[["text", "label"]].sample(min(5, count))
                for idx, row in sample_comments.iterrows():
                    st.text(f"[{row['label']}] {row['text']}")
                    
        else:
            st.warning(f"Kata '{search_keyword}' tidak ditemukan dalam dataset.")
