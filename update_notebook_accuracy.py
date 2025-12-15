import json

file_path = 'd:/SEMESTER 7/NLP/uas tubes analisis sentimen/AS-uas-klmpk4.ipynb'

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    modified_count = 0
    
    # New code content for TF-IDF and Model Training cells
    # We will look for the cell that does TF-IDF and Model Training
    
    new_tfidf_code = [
        "# --- Text Vektorisasi (TF-IDF) ---\n",
        "# Mengubah kata-kata menjadi angka agar dimengerti mesin\n",
        "# Gunakan ngram_range=(1,2) untuk menangkap frasa kata\n",
        "tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))\n",
        "X = tfidf.fit_transform(df_clean['cleaned_text'])\n",
        "y = df_clean['label']\n",
        "\n",
        "# --- Data Splitting ---\n",
        "# Membagi data latih (80%) dan data uji (20%)\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
        "\n",
        "# --- OVERSAMPLING MANUAL ---\n",
        "from sklearn.utils import resample\n",
        "\n",
        "# Gabungkan X_train dan y_train\n",
        "train_data = pd.DataFrame(X_train.toarray(), columns=tfidf.get_feature_names_out())\n",
        "train_data['label'] = y_train.values\n",
        "\n",
        "# Pisahkan berdasarkan kelas\n",
        "df_netral = train_data[train_data['label'] == 'Netral']\n",
        "df_positif = train_data[train_data['label'] == 'Positif']\n",
        "df_negatif = train_data[train_data['label'] == 'Negatif']\n",
        "\n",
        "# Oversample minoritas\n",
        "n_samples = len(df_netral)\n",
        "if not df_positif.empty and n_samples > 0:\n",
        "    df_positif_os = resample(df_positif, replace=True, n_samples=n_samples, random_state=42)\n",
        "else:\n",
        "    df_positif_os = df_positif\n",
        "\n",
        "if not df_negatif.empty and n_samples > 0:\n",
        "    df_negatif_os = resample(df_negatif, replace=True, n_samples=n_samples, random_state=42)\n",
        "else:\n",
        "    df_negatif_os = df_negatif\n",
        "\n",
        "# Gabungkan kembali\n",
        "df_train_balanced = pd.concat([df_netral, df_positif_os, df_negatif_os])\n",
        "\n",
        "# Pisahkan X dan y lagi\n",
        "X_train_balanced = df_train_balanced.drop('label', axis=1).values\n",
        "y_train_balanced = df_train_balanced['label']\n",
        "\n",
        "print(f'\\nJumlah Data Latih (Balanced): {X_train_balanced.shape[0]}')\n",
        "print(f'Jumlah Data Uji: {X_test.shape[0]}')"
    ]

    new_model_code = [
        "# ==========================================\n",
        "# 4. DATA MODELING\n",
        "# ==========================================\n",
        "\n",
        "# Import LogisticRegression\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "\n",
        "# Kita ganti ke Logistic Regression karena lebih robust untuk text classification\n",
        "model = LogisticRegression(max_iter=1000, random_state=42)\n",
        "model.fit(X_train_balanced, y_train_balanced)\n",
        "print(\"\\nModel Logistic Regression berhasil dilatih!\")"
    ]

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            source_text = "".join(source)
            
            # Update TF-IDF Cell
            # Look for lines characteristic of the old TF-IDF block
            if "tfidf = TfidfVectorizer(max_features=1000)" in source_text or "X = tfidf.fit_transform(df_clean['cleaned_text'])" in source_text:
                if "OVERSAMPLING MANUAL" not in source_text: # Validate it's not already updated
                     # Find the cell that has splitting logic
                     if "train_test_split" in source_text:
                        cell['source'] = new_tfidf_code
                        modified_count += 1
            
            # Update Modeling Cell
            if "model = MultinomialNB()" in source_text:
                cell['source'] = new_model_code
                modified_count += 1

    if modified_count > 0:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Notebook updated successfully. Modified {modified_count} cells.")
    else:
        print("No matching cells found to replace or already updated.")

except Exception as e:
    print(f"Error updating notebook: {e}")
