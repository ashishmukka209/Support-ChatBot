# cdp_chatbot_hf_final_updated.py

import os
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

###############################################
# 1. ENVIRONMENT SETUP
###############################################
# Set your Hugging Face API token (obtain one from https://huggingface.co/settings/tokens)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "*****"  # replace with your token

###############################################
# 2. DOCUMENTATION URLs
###############################################
SEGMENT_DOCS_URLS = ["https://segment.com/docs/"]
MPARTICLE_DOCS_URLS = ["https://docs.mparticle.com/"]
LYTICS_DOCS_URLS = ["https://docs.lytics.com/"]
ZEOTAP_DOCS_URLS = ["https://docs.zeotap.com/home/en-us/"]

###############################################
# 3. SCRAPING FUNCTIONS
###############################################
def scrape_website_text(url):
    """
    Fetch HTML from the URL and return its visible text.
    Note: This is a basic scraper. For production, consider targeting the main content only.
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch {url} -> {e}")
        return ""
    
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text(separator="\n")

def gather_all_docs():
    """Collect text from all provided documentation URLs."""
    all_text = ""
    
    print("Scraping Segment Docs...")
    for url in SEGMENT_DOCS_URLS:
        all_text += scrape_website_text(url) + "\n"
    
    print("Scraping mParticle Docs...")
    for url in MPARTICLE_DOCS_URLS:
        all_text += scrape_website_text(url) + "\n"
    
    print("Scraping Lytics Docs...")
    for url in LYTICS_DOCS_URLS:
        all_text += scrape_website_text(url) + "\n"
    
    print("Scraping Zeotap Docs...")
    for url in ZEOTAP_DOCS_URLS:
        all_text += scrape_website_text(url) + "\n"
    
    return all_text

###############################################
# 4. BUILD THE VECTOR STORE
###############################################
def build_vectorstore_from_docs(docs_text):
    # Increase chunk size and overlap to capture more complete context.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    doc_splits = text_splitter.create_documents([docs_text])
    
    # Use a Hugging Face embedding model (all-MiniLM-L6-v2) for efficient embeddings.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Build an in-memory vector store with Chroma.
    vectorstore = Chroma.from_documents(doc_splits, embeddings, collection_name="cdp-docs")
    return vectorstore

###############################################
# 5. BUILD THE RETRIEVAL QA CHAIN
###############################################
def build_qa_chain(vectorstore):
    # Use a Hugging Face model for inference; here we try a larger model for better answers.
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",  # Change to flan-t5-xl if resources allow
        model_kwargs={"temperature": 0.0, "max_length": 512}
    )
    
    # Retrieve more chunks (k=5) for better context.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Build the RetrievalQA chain with default prompt settings (chain_type="stuff")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True  # Enable to debug retrieved content
    )
    return qa_chain

###############################################
# 6. MAIN CHAT LOOP
###############################################
def main():
    print("Gathering documentation text (this may take a while)...")
    docs_text = gather_all_docs()
    
    print("Building vector store...")
    vectorstore = build_vectorstore_from_docs(docs_text)
    
    print("Building QA chain...")
    qa_chain = build_qa_chain(vectorstore)
    
    print("\nCDP Chatbot is ready! Ask your question (type 'exit' to quit)\n")
    
    while True:
        user_query = input("You: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Exiting. Goodbye!")
            break
        
        # Get answer from the retrieval QA chain.
        response = qa_chain({"query": user_query})
        answer = response["result"]
        
        # Debug: Print snippet of each retrieved document (optional).
        source_docs = response.get("source_documents", [])
        print("DEBUG: Retrieved document snippets:")
        for idx, doc in enumerate(source_docs):
            print(f"Snippet {idx+1}: {doc.page_content[:200]}...\n")
        
        print("CDP Assistant:", answer, "\n")

if __name__ == "__main__":
    main()
