import json
nb_path = 'd:/SEMESTER 7/NLP/uas tubes analisis sentimen/AS-uas-klmpk4.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source_snippet = "".join(cell['source'])[:100].replace('\n', ' ')
        print(f"Cell {i}: {source_snippet}...")
