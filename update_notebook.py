import json

file_path = 'd:/SEMESTER 7/NLP/uas tubes analisis sentimen/AS-uas-klmpk4.ipynb'

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    modified_count = 0
    # Iterate through cells to find the one with labeling logic
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            new_source = []
            cell_modified = False
            for line in source:
                if "df_clean['label'] = df_clean['cleaned_text'].apply(get_sentiment)" in line:
                    # Replace the line with conditional logic
                    new_source.append("    if 'label' in df_clean.columns:\n")
                    new_source.append("        print('Menggunakan label yang sudah ada dari dataset.')\n")
                    new_source.append("    else:\n")
                    new_source.append("        # Fallback to TextBlob\n")
                    new_source.append("        df_clean['label'] = df_clean['cleaned_text'].apply(get_sentiment)\n")
                    cell_modified = True
                else:
                    new_source.append(line)
            
            if cell_modified:
                cell['source'] = new_source
                modified_count += 1

    if modified_count > 0:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Notebook updated successfully. Modified {modified_count} cells.")
    else:
        print("No matching line found to replace.")

except Exception as e:
    print(f"Error updating notebook: {e}")
