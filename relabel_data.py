import pandas as pd
import re
from textblob import TextBlob

# Load the dataset
file_path = 'dataset_tiktok-comments-scraper_2025-12-09_02-52-00-697.csv'
try:
    df = pd.read_csv(file_path)
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Define keywords
positive_words = [
    "ganteng", "pinter", "semangat", "keren", "idaman", "trust", "love", "suka", "sayang", "baikk",
    "joss", "mantap", "aamiin", "sukses", "best", "bagus", "menarik", "sehat", "bahagia", "top",
    "tier", "aman", "jaya", "menyala", "gas", "sikat", "masya allah", "subhanallah", "alhamdulillah",
    "smart", "bijak", "lucu", "ngakak", "wkwk", "haha", "hebat", "pro", "suhu", "terbaik", "idola",
    "senyum", "ketawa", "kocak", "menghibur", "seru", "asik", "enak", "nyaman", "adem", "tenang",
    "damai", "rindu", "kangen", "cinta", "beauty", "cantik", "manis", "imut", "cute", "wow", "kerenn", "bismillah"
]

negative_words = [
    "takut", "buaya", "redflag", "jahat", "toxic", "sampah", "jelek", "benci", "marah", "kesel",
    "kecewa", "bosen", "capek", "lelah", "pusing", "bingung", "susah", "sulit", "ribet", "mahal",
    "rugi", "bohong", "palsu", "curang", "anjing", "babi", "bangsat", "goblok", "tolol", "bodoh",
    "gila", "setan", "sakit", "sedih", "nangis", "kecewa", "hancur", "rusak", "parah", "buruk",
    "stress", "stres", "gagal", "kalah", "batal", "tumbang", "trauma", "hindari", "jangan", "run", "kabur"
]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_sentiment(text):
    clean = clean_text(text)
    
    # Check for direct matches
    for word in positive_words:
        if word in clean.split():
            return "Positif"
    for word in negative_words:
        if word in clean.split():
            return "Negatif"
            
    # Fallback to TextBlob for English or other patterns
    analysis = TextBlob(str(text))
    if analysis.sentiment.polarity > 0.05:
        return "Positif"
    elif analysis.sentiment.polarity < -0.05:
        return "Negatif"
    else:
        return "Netral"

# Apply labeling
df['label'] = df['text'].apply(get_sentiment)

# Save the new dataset
output_path = 'dataset_tiktok-comments-scraper_2025-12-09_02-52-00-697.csv' # Overwriting as per flow, or can create new
df.to_csv(output_path, index=False)

# Print stats
print("Re-labeling complete.")
print("Distribution:")
print(df['label'].value_counts())
