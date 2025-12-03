import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="Semantic Book Recommender", layout="wide")
load_dotenv()

DB_DIR = "./chroma_db"

# --- 2. CSS STYLING (For consistent card sizes) ---
st.markdown("""
    <style>
    div[data-testid="stImage"] img {
        height: 250px;
        object-fit: cover;
        border-radius: 8px;
    }
    div[data-testid="stExpander"] {
        border: none;
        box-shadow: none;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. LOAD DATA ---
@st.cache_data
def load_book_data():
    try:
        df = pd.read_csv("books_with_emotions.csv")
        
        # Fix Thumbnails
        df["large_thumbnail"] = df["thumbnail"] + "&fife=w800"
        df["large_thumbnail"] = np.where(
            df["large_thumbnail"].isna(),
            "https://via.placeholder.com/150x200?text=No+Cover",
            df["large_thumbnail"],
        )
        return df
    except FileNotFoundError:
        st.error("❌ 'books_with_emotions.csv' not found in the current directory.")
        st.stop()

# --- 4. SETUP VECTOR DATABASE ---
@st.cache_resource
def setup_vector_db():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    if os.path.exists(DB_DIR) and os.path.isdir(DB_DIR):
        print("Loading existing Vector DB from disk...")
        db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    else:
        print("Creating new Vector DB from CSV column 'tagged_description'...")
        
        # Load raw data to build the DB
        df = pd.read_csv("books_with_emotions.csv")
        
        if "tagged_description" not in df.columns:
            st.error("❌ The CSV is missing the 'tagged_description' column.")
            st.stop()
            
        documents = []
        for _, row in df.iterrows():
            # CRITICAL: We put the ISBN first because the search logic splits the string 
            # and takes the first word to identify the book.
            # Format: "ISBN Tagged_Description"
            content = f"{row['isbn13']} {row['tagged_description']}"
            documents.append(Document(page_content=content))
            
        db = Chroma.from_documents(documents, embeddings, persist_directory=DB_DIR)
        
    return db

# --- INITIALIZATION ---
with st.spinner("Initializing AI Models and Database..."):
    try:
        books = load_book_data()
        db_books = setup_vector_db()
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        st.stop()

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# --- 5. SEARCH ALGORITHM ---
def retrieve_semantic_recommendations(query, category, tone, initial_top_k=50, final_top_k=16):
    
    # 1. Semantic Search via Chroma/Ollama
    recs = db_books.similarity_search(query, k=initial_top_k)
    
    # 2. Extract ISBNs (First token of the page_content)
    books_list = []
    for rec in recs:
        try:
            # content looks like: "9780141... The description..."
            isbn_str = rec.page_content.strip('"').split()[0] 
            books_list.append(int(isbn_str))
        except ValueError:
            continue # Skip if ISBN parsing fails

    # 3. Filter Metadata in Pandas
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    # 4. Sort by Tone
    tone_map = {
        "Happy": "joy", "Surprising": "surprise", "Angry": "anger", 
        "Suspenseful": "fear", "Sad": "sadness"
    }
    if tone in tone_map:
        book_recs.sort_values(by=tone_map[tone], ascending=False, inplace=True)

    return book_recs

# --- 6. UI LAYOUT ---
st.title("📚 Next Book AI")
st.markdown("### Semantic Search & Emotion-Based Recommendations")

# Sidebar
with st.sidebar:
    st.header("Filters")
    selected_category = st.selectbox("Category", categories)
    selected_tone = st.selectbox("Emotional Tone", tones)
    st.divider()
    st.info("Powered by Ollama (nomic-embed-text) & ChromaDB")

# Search Bar
col1, col2 = st.columns([4, 1])
with col1:
    user_query = st.text_input("Describe your ideal book...", placeholder="e.g. A story about a robot learning to love")
with col2:
    # Add some vertical spacing to align button
    st.markdown("<br>", unsafe_allow_html=True) 
    search_btn = st.button("Find Books", type="primary", use_container_width=True)

# Logic Execution
if search_btn or user_query:
    if not user_query:
        st.warning("Please enter a description.")
    else:
        with st.spinner("Reading books..."):
            try:
                results = retrieve_semantic_recommendations(user_query, selected_category, selected_tone)
                
                if results.empty:
                    st.info("No matching books found.")
                else:
                    # Results Grid
                    cols = st.columns(4)
                    for idx, (_, row) in enumerate(results.iterrows()):
                        with cols[idx % 4]:
                            with st.container(border=True):
                                st.image(row["large_thumbnail"], use_container_width=True)
                                st.markdown(f"**{row['title']}**")
                                
                                # Format Authors
                                authors = str(row["authors"]).split(";")
                                auth_text = authors[0] + " et al." if len(authors) > 1 else authors[0]
                                st.caption(auth_text)
                                
                                with st.expander("Synopsis"):
                                    st.write(row["description"])
            except Exception as e:
                st.error(f"Error during search: {e}")
