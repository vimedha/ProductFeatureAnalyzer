import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from keybert import KeyBERT
from textblob import TextBlob
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import re
import contractions
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import unicodedata
from nltk.stem import WordNetLemmatizer
import html


# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Setup
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

st.set_page_config(page_title="🧠 Customer Review Analyzer", layout="wide")
st.title("🧠 Customer Review Analyzer")

# File Upload
uploaded_file = st.file_uploader("📤 Upload your customer reviews CSV file", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

    st.success("✅ CSV uploaded successfully!")
    st.write("📄 Preview of your data:")
    st.dataframe(df.head())

    if "ProductName" not in df.columns or "Review" not in df.columns:
        st.error("❌ Columns 'ProductName' and 'Review' must be in your dataset.")
    else:
        # Step: Brand Detection
        st.subheader("🔍 Detected Business/Brand Names (via Zero-Shot)")

        @st.cache_resource
        def load_zero_shot():
            return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

        zero_shot = load_zero_shot()

        possible_brands = df["ProductName"].astype(str).apply(lambda x: x.split()[0].strip().capitalize())
        possible_brands = [b for b in possible_brands if b.lower() not in stop_words]
        possible_brands = sorted(set([b.lower().capitalize() for b in possible_brands]))

        label = "brand or company name"
        brand_candidates = []
        with st.spinner("Classifying potential brand names..."):
            for brand in possible_brands:
                result = zero_shot(brand, candidate_labels=[label])
                if result["scores"][0] > 0.7:
                    brand_candidates.append(brand)

        if brand_candidates:
            selected_brand = st.selectbox("✅ Select a detected brand:", brand_candidates)
            st.success(f"You selected: **{selected_brand}**")

            # Step: Product Selection
            st.subheader("📦 Products Offered by Selected Brand")
            brand_products = df[df["ProductName"].str.lower().str.startswith(selected_brand.lower())].copy()

            def extract_product_name(row):
                tokens = row.split()
                return " ".join(tokens[1:]).strip() if len(tokens) > 1 else "Generic Product"

            brand_products["ProductTitle"] = brand_products["ProductName"].apply(extract_product_name)
            unique_products = sorted(set(brand_products["ProductTitle"]))

            if not unique_products:
                st.warning("⚠️ No products found for the selected brand.")
            else:
                selected_product = st.selectbox("🛒 Choose a product under this brand", unique_products)
                st.success(f"Selected product: **{selected_product}**")

                # Step: Feature Extraction
                st.subheader("🧠 Key Features Mentioned by Customers")

                target_reviews = brand_products[brand_products["ProductTitle"] == selected_product].copy()
                if target_reviews.empty:
                    st.warning("⚠️ No reviews found for this product.")
                else:
                    @st.cache_resource
                    def load_flan_model():
                        model_name = "google/flan-t5-large"
                        tokenizer = AutoTokenizer.from_pretrained(model_name)
                        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        model.to(device)
                        return tokenizer, model, device

                    tokenizer, flan_model, device = load_flan_model()
                    kw_model = KeyBERT()

                    def clean_text(text):
                        text = html.unescape(text)
                        text = BeautifulSoup(text, "html.parser").get_text()
                        text = contractions.fix(text)
                        text = re.sub(r"http\S+|www\S+", "", text)
                        text = re.sub(r'\S+@\S+', '', text)
                        text = re.sub(r"\d+", "", text)
                        text = unicodedata.normalize("NFKD", text)
                        text = text.encode("ascii", "ignore").decode("utf-8", "ignore")
                        text = re.sub(r"[^\w\s]", "", text)
                        text = text.lower()
                        tokens = text.split()
                        tokens = [word for word in tokens if word not in stop_words]
                        tokens = [lemmatizer.lemmatize(word) for word in tokens]
                        return " ".join(tokens)

                    target_reviews["cleaned_review"] = target_reviews["Review"].astype(str).apply(clean_text)

                    def extract_keybert_keywords(text):
                        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=5)
                        return ", ".join([kw[0] for kw in keywords])

                    def flan_enhance_features(keywords):
                        if not keywords.strip():
                            return ""
                        prompt = (
                            "Here are some product reviews and the key features mentioned in them:\n\n"
                            "Review: 'This phone has a great battery life and a sharp display.'\n"
                            "Features: battery life, display\n\n"
                            "Review: 'The headphones are comfortable and the bass is impressive.'\n"
                            "Features: comfort, bass quality\n\n"
                            f"Review: '{keywords}'\n"
                            "Features:"
                        )
                        inputs = tokenizer(prompt, return_tensors="pt").to(device)
                        outputs = flan_model.generate(**inputs, max_new_tokens=60)
                        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

                    def extract_features_pipeline(text):
                        raw_keywords = extract_keybert_keywords(text)
                        return flan_enhance_features(raw_keywords)

                    sample_df = target_reviews.sample(min(10, len(target_reviews)), random_state=42).copy()
                    sample_df["Extracted Features"] = sample_df["cleaned_review"].apply(extract_features_pipeline)

                    st.write("🔬 Sample Extracted Features")
                    st.dataframe(sample_df[["Review", "Extracted Features"]])

                    # Step: Sentiment Analysis
                    st.subheader("📊 Sentiment Overview")

                    def get_sentiment(text):
                        blob = TextBlob(text)
                        polarity = blob.sentiment.polarity
                        if polarity > 0.1:
                            return "Positive"
                        elif polarity < -0.1:
                            return "Negative"
                        else:
                            return "Neutral"

                    target_reviews["Sentiment"] = target_reviews["Review"].astype(str).apply(get_sentiment)
                    sentiment_counts = target_reviews["Sentiment"].value_counts()

                    fig, ax = plt.subplots(figsize=(5, 5))
                    colors = {"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#95a5a6"}
                    sentiment_counts.plot.pie(
                        ax=ax,
                        autopct="%1.1f%%",
                        startangle=90,
                        colors=[colors.get(label, "#3498db") for label in sentiment_counts.index],
                        labels=sentiment_counts.index,
                        wedgeprops=dict(width=0.5)
                    )
                    ax.set_ylabel("")
                    ax.set_title("Customer Sentiment Distribution", fontsize=14)
                    st.pyplot(fig)

                    st.markdown("### 📝 Sample Reviews with Sentiment")
                    st.dataframe(target_reviews[["Review", "Sentiment"]].head(10))
        else:
            st.warning("⚠️ No confident brand names detected.")
else:
    st.info("👈 Upload a CSV file to get started.")
