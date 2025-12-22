import json

nb_path = 'd:/SEMESTER 7/NLP/uas tubes analisis sentimen/AS-uas-klmpk4.ipynb'

try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Update Imports (Cell 0)
    cell0 = nb['cells'][0]
    updated_source = []
    for line in cell0['source']:
        updated_source.append(line)
    
    str_source = "".join(updated_source)
    if "LogisticRegression" not in str_source:
        updated_source.insert(len(updated_source)-1, "from sklearn.linear_model import LogisticRegression\n")
    if "accuracy_score" not in str_source:
        updated_source.insert(len(updated_source)-1, "from sklearn.metrics import accuracy_score\n")
    
    cell0['source'] = updated_source

    # Replace Modeling Cell (Cell 4)
    cell4 = nb['cells'][4]
    
    new_model_source = [
        "# ==========================================\n",
        "# 4. DATA MODELING & COMPARISON\n",
        "# ==========================================\n",
        "\n",
        "# 1. Naive Bayes\n",
        "print(\"Training Naive Bayes...\")\n",
        "nb_model = MultinomialNB()\n",
        "nb_model.fit(X_train_balanced, y_train_balanced)\n",
        "y_pred_nb = nb_model.predict(X_test)\n",
        "acc_nb = accuracy_score(y_test, y_pred_nb)\n",
        "\n",
        "# 2. Logistic Regression (Model pada App)\n",
        "print(\"Training Logistic Regression...\")\n",
        "lr_model = LogisticRegression(max_iter=1000, random_state=42)\n",
        "lr_model.fit(X_train_balanced, y_train_balanced)\n",
        "y_pred_lr = lr_model.predict(X_test)\n",
        "acc_lr = accuracy_score(y_test, y_pred_lr)\n",
        "\n",
        "# --- PERBANDINGAN ---\n",
        "print(f\"\\nAkurasi Naive Bayes: {acc_nb:.4f}\")\n",
        "print(f\"Akurasi Logistic Regression: {acc_lr:.4f}\")\n",
        "\n",
        "if acc_lr > acc_nb:\n",
        "    print(\"\\nKESIMPULAN: Logistic Regression lebih akurat.\")\n",
        "else:\n",
        "    print(\"\\nKESIMPULAN: Naive Bayes lebih akurat atau sama.\")\n",
        "\n",
        "print(\"\\nClassification Report (Logistic Regression):\")\n",
        "print(classification_report(y_test, y_pred_lr))"
    ]
    
    nb['cells'][4]['source'] = new_model_source

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated successfully.")

except Exception as e:
    print(f"Error: {e}")
