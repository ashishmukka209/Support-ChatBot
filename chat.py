import os
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA


os.environ["HUGGINGFACEHUB_API_TOKEN"] = "*****"  


SEGMENT_DOCS_URLS = ["https://segment.com/docs/"]
MPARTICLE_DOCS_URLS = ["https://docs.mparticle.com/"]
LYTICS_DOCS_URLS = ["https://docs.lytics.com/"]
ZEOTAP_DOCS_URLS = ["https://docs.zeotap.com/home/en-us/"]

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


def build_vectorstore_from_docs(docs_text):
   
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    doc_splits = text_splitter.create_documents([docs_text])
    
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
   
    vectorstore = Chroma.from_documents(doc_splits, embeddings, collection_name="cdp-docs")
    return vectorstore


def build_qa_chain(vectorstore):
    
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",  
        model_kwargs={"temperature": 0.0, "max_length": 512}
    )
    
   
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
   
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True  
    )
    return qa_chain


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
        
        
        response = qa_chain({"query": user_query})
        answer = response["result"]
        
       
        source_docs = response.get("source_documents", [])
        print("DEBUG: Retrieved document snippets:")
        for idx, doc in enumerate(source_docs):
            print(f"Snippet {idx+1}: {doc.page_content[:200]}...\n")
        
        print("CDP Assistant:", answer, "\n")

if __name__ == "__main__":
    main()
