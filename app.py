# Creator: Ikenna Oluigbo
# Contact me on callme-smartai@proton.me 
# Program to analyze document, summarize, and provide chat responses, suggest related articles online, and download images from documents

#Importing Libraries 
import fitz
import pdfplumber
from openai import OpenAI
import os
import chromadb
import streamlit as st
from streamlit_js_eval import streamlit_js_eval
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction, OllamaEmbeddingFunction
import time

import tempfile
import shutil
import uuid
from pathlib import Path


OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]


# ---- Config ----
BASE_TMP_DIR = Path(tempfile.gettempdir()) / "myapp_sessions"
PERMANENT_DIR = Path("permanent_uploads")  # where finished files end up
MAX_SESSION_AGE_SECONDS = 0.5 * 60 * 60  # delete abandoned temp folders after 1 hour

BASE_TMP_DIR.mkdir(parents=True, exist_ok=True)
PERMANENT_DIR.mkdir(parents=True, exist_ok=True)


class LLMModel:
    def __init__(self, model_type = "openai"): 
        self.model_type = model_type 
        if model_type == "openai":
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.model_name = "gpt-5.4"
        else:
            self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            self.model_name = "llava"
        
    def generate_completion(self, message):
        try:
            response = self.client.chat.completions.create(
                messages=message,
                model=self.model_name             
                )
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error generating content: {str(e)}"


class EmbeddingModel:
    def __init__(self, model_type="openai"):
        self.model_type = model_type
        if model_type == "openai":
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.embedding_fn = OpenAIEmbeddingFunction(
                api_key=OPENAI_API_KEY,
                model_name="text-embedding-3-small", 
                dimensions=1536
            )
        
        else: 
            model_type == "ollama_Nomic"
            self.embedding_fn = OpenAIEmbeddingFunction(
                api_key="ollama",
                api_base="http://localhost:11434/v1",
                model_name="nomic-embed-text",
                dimensions=768
            )       


def cleanup_stale_sessions(max_age_seconds: int = MAX_SESSION_AGE_SECONDS):
    """Remove session folders that are older than max_age_seconds.
    Call this at app startup so abandoned folders eventually get swept."""
    now = time.time()
    for folder in BASE_TMP_DIR.iterdir():
        if folder.is_dir():
            age = now - folder.stat().st_mtime
            if age > max_age_seconds:
                shutil.rmtree(folder, ignore_errors=True)


def get_session_dir() -> Path:
    """Create (once per session) a unique temp folder for this user."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    session_dir = BASE_TMP_DIR / st.session_state.session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def move_to_permanent_storage(session_dir: Path):
    """Move all files from the session temp folder to permanent storage,
    then delete the now-empty temp folder."""
    dest = PERMANENT_DIR / st.session_state.session_id
    dest.mkdir(parents=True, exist_ok=True)

    for file_path in session_dir.iterdir():
        shutil.move(str(file_path), str(dest / file_path.name))

    shutil.rmtree(session_dir, ignore_errors=True)
    return dest


# Function to split text
def make_chunks(texts : str, pdf_file, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Splitting the texts into chunks"""
    
    chunks = []
    start = 0
    while start < len(texts):
        end = start + chunk_size
        
        #If not beginning from start, include overlap
        if start > 0:
            start = start - chunk_overlap
        
        chunk = texts[start:end]
        
        #Try to break at end of sentence to preserve full paragraph context
        if end < len(texts):
            last_period = chunk.rfind(".")
            if last_period != -1:
                chunk = chunk[ : last_period + 1]
                end = start + last_period + 1
                
        
        chunks.append(
            {"id" : str(uuid.uuid4()),
             "text" : chunk,
             "metadata" : {"source": pdf_file.name}
             }
        )
        
        start = end
    
    return chunks


# Function to parse pdf texts
def read_pdfuploaded_text(pdfupload_path):
    temp_doc = ""
    #for pdf in os.listdir(pdfupload_path):
    doc = fitz.open(pdfupload_path)
    print(f"Total lenght of Document Pages: {len(doc)} ")
    for i in range(len(doc)):
        curr_page = doc[i]
        temp_doc = temp_doc + curr_page.get_text() + "\n"
    
    return temp_doc


#Extract Title
def read_pdf_title(pdfupload_path):
    #Method 1: Extract from Metadata
    
    with pdfplumber.open(pdfupload_path) as p:
        page = p.pages[0]
        # Filter characters with a size greater than a threshold (e.g., 10)
        # Adjust threshold based on your PDF's font sizes
        filtered_page = page.filter(lambda x: x.get("size", 0) > 12)
        title = filtered_page.extract_text()
        print(f"Heuristic Title: {title}")
    
    return title


#Image Extraction        
def read_pdf_image(pdfupload_path):
    temp_bytes = [] ; temp_ext = []
    
    doc = fitz.open(pdfupload_path)
    for page_num in range(len(doc)):   #Loop through all pages and find images from images (jpeg, jpg, png. OCR images cant be seen)
        page = doc[page_num]
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
        
            temp_bytes.append(image_bytes)
            temp_ext.append(image_ext)
            
    return temp_bytes, temp_ext
        
        
#Split Text and assign ID
# def split_text_id(docs):
#     messages_with_id = dict()
#     chunks = make_chunks(texts=docs, chunk_size=1000, chunk_overlap=200)
#     for i, chunk in enumerate(chunks):
#         idx = i + 1
#         messages_with_id[str(idx)] = chunk
    
#     return messages_with_id


def setup_chromadb(texts, embedding_model):
    collection_name = "doc_analyzer"
    db_path = "./ChromaDB/llm_docdb"
    
    client = chromadb.PersistentClient(path=db_path)
  
    collection = client.get_or_create_collection(
        name=collection_name, 
        embedding_function=embedding_model.embedding_fn
    )
        
    # ids = list(texts.keys())
    # documents = list(texts.values())
    
    for doc in texts:
    
        collection.add(ids=doc["id"],
                      documents=doc["text"],
                    metadatas=doc["metadata"])
   
    print("\nDocuments added to ChromaDB collection successfully!")
    return collection


def find_related_chunks(query, collection, top_k=5):
    results = collection.query(query_texts=[query], n_results=top_k)
    #print(results)
    return results["documents"][0]
    

def augment_prompt(query, relevant_chunks):
    context = "\n".join([chunks for chunks in relevant_chunks])
    #print(relevant_chunks)
    augmented_prompt = f"Context: \n{context} \n\nQuestion: {query}:"
    
    print("Augmented prompt: ⤵️")
    print(augmented_prompt)
    
    return augmented_prompt


def rag_pipeline(query, collection, llm_model, top_k=5):
    print(f"\nNow processing query: {query}")
    
    relevant_chunks = find_related_chunks(query, collection, top_k)
    augmented_prompt = augment_prompt(query, relevant_chunks)
    
    response = llm_model.generate_completion(
        [
        {"role": "system", "content": f"""You are a very smart assistant.
                    You can are very knowledgeable and can effectively answer any question from within the context information, or just about any question. 
                    For each user associated with a session ID, you should ONLY answer their questions as it is directly related to their uploaded docuemtns. 
                    Do not consider all previous context in your knowledge base for each user session. Only reference the context information from uploaded documents.
                    To answer questions, first rigorously consult the context information in {augmented_prompt} related to the {query} as per user's uploaded document.
                    Let your response be based only on the context information in {augmented_prompt}. Restrict your response on to documents uploaded in each session.
                    If you are not sure, ask the user to give you keywords from uploaded documents. That way you can restrict your responses only to context information from that uploaded documents in each session.
                    Occassionally, if you are asked questions outside the context, first try to answer based on context information in {augmented_prompt}.
                    If the answer is not in the context information, you may then use your base knowledge to answer the question. 
                    When you use your base knowledge, inform the user that the answer is not found in context information and that you are pulling response from your knowledge base
                    
                    ### Polite Response:
                    In your response, consider including:
                    - Acknowledge the user’s query and express gratitude for the opportunity to assist.
                    - Provide a clear and concise answer that directly addresses the question, based on context information in {augmented_prompt}. Ensure no ambiguity or hallucination.
                    - Use positive language and maintain a supportive tone throughout.
                    - Use context information from only documents uploaded in a user session. 
                    - If applicable, include relevant information or resources that could help further. Only when neccessary
                    - Conclude by inviting any follow-up questions or providing encouragement for the user’s pursuit of information."""},
        
        {"role": "user", "content": query}
        ]
    )
    
    print("\nGenerated response:")
    print(response)

    references = [chunk for chunk in relevant_chunks]
    return response, references

all_documents = [] 

def streamlit_app():
    cleanup_stale_sessions()
    
    st.set_page_config(page_title="Smart Research Article Analyzer", layout="wide")
    st.title("🧾 Smart Article: Document Analyzer with RAG Integration")

    st.caption("How can Smart Article help you? Just Upload your document(s) and Smart Article will; ")
    st.write()
    st.badge("- Analyze your document(s) and embed your document(s) in a vector store through series of structured processes")
    st.badge("- Provide semantic search & Maximal Marginal Relevance functionality through querying Smart Article, providing smart responses. Just ask whatever query, and you get a response!")
    st.badge("- Responses are entirely based on the uploaded document(s), references for the responses are also provided")
    st.badge("- Crawl the web and provide up to 20 other articles related to the document(s) uploaded, with download links. Just use the keyword: papers")
    st.badge("- Extract all images from the document(s) uploaded, providing a download button. Just use the keyword: images")
    st.badge("- Provide the opportunity to choose between different Model and Embedding Architectures")

    # Sidebar for model selection
    st.sidebar.title("Model Configuration")

    llm_type = st.sidebar.radio(
        "Select LLM Model:",
        ["openai", "ollama"],
        format_func=lambda x: "OpenAI GPT (Most Stable)" if x == "openai" else "Ollama llava",
    )

    embedding_type = st.sidebar.radio(
        "Select Embedding Model:",
        ["openai", "ollama_Nomic"],
        format_func=lambda x: {
            "openai": "OpenAI Embeddings (Most Stable)",
            "ollama_Nomic": "Nomic Embed Text (Ollama)",
        }[x],
    )


    st.sidebar.write(
        f"\nCurrent Embedding Model:\n",
        f"- Name: {" ".join(embedding_type.split("_")).upper()}\n",
        f"- Dimensions: {EmbeddingModel(embedding_type).embedding_fn.dimensions}"
    )
    
    st.sidebar.image("./llm_bot.jpg")

    
    #Adding Footer text
    footer = """
    <style>
    a:link, a:visited {
        color: blue;
        text-decoration: underline;
    }
    
    a:hover, a:active {
        color: red;
        text-decoration: underline;
    }
    
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: white;
        color: black;
        text-align: center;
        padding: 10px;
        z-index: 9999;
    }
    </style>
    
    <div class="footer">
        <p>
            Developed by
            <a href="https://github.com/ikenna-oluigbo/" target="_blank">
                Ikenna Oluigbo (PhD).
            </a>
            Drop a feedback!
        </p>
    </div>
    """

    st.markdown(footer, unsafe_allow_html=True)
    

    #Beginning PDF RAG Ingestion
    
    session_dir = get_session_dir()
    st.caption(f"Session ID: {st.session_state.session_id}")
    
    st.write("#PDF Ingestion - RAG Knowledge Base ")

    if "pdf_docs" not in st.session_state:
        st.session_state.pdf_docs = []


    try:
        uploaded_files = st.file_uploader("""Upload ONLY PDF Files. You can upload one or more files (Upload within 20 seconds)""", 
                                        type="pdf", 
                                        accept_multiple_files=True,
                                        max_upload_size=15)    #MB
    
            
        if uploaded_files:
            for uploaded_file in uploaded_files:
                save_path = session_dir / uploaded_file.name
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.success(f"Saved {len(uploaded_files)} file(s) to your private session folder.")
            
            
            # --- Do your PDF analysis here, reading from session_dir ---
            all_titles = list() 
            for file_path in session_dir.iterdir():
                st.write(f"Analyzing: {file_path.name}")
                temp_doc = read_pdfuploaded_text(file_path)
                
                chunks = make_chunks(texts=temp_doc, pdf_file=file_path, chunk_size=1000, chunk_overlap=200)
                all_documents.extend(chunks)
                
                t = read_pdf_title(file_path)
                all_titles.append(t)  
            
             
        time.sleep(23)

        st.write(f"All {len(uploaded_files)} files uploaded successfully! ✅") 
        
        # analyzed_documents = " ".join(all_documents)            #Merge all strings into one
    
    
        # Initialize session state
        if "initialized" not in st.session_state:
            st.session_state.initialized = False    
            
            st.session_state.facts = all_documents
    
            # Initialize models
            st.session_state.llm_model = LLMModel(llm_type)
            st.session_state.embedding_model = EmbeddingModel(embedding_type)
    
            # Setup ChromaDB
            documents = st.session_state.facts         #Input text is a dictionary with keys and values
            st.session_state.collection = setup_chromadb(
                documents, st.session_state.embedding_model
            )
            st.session_state.initialized = True
    
        # If models changed, reinitialize
        if (
              
            st.session_state.llm_model.model_type != llm_type
            or st.session_state.embedding_model.model_type != embedding_type
        ):
            st.session_state.llm_model = LLMModel(llm_type)
            st.session_state.embedding_model = EmbeddingModel(embedding_type)
            documents = st.session_state.facts
            st.session_state.collection = setup_chromadb(
                documents, st.session_state.embedding_model
            )
    
        # Display uploaded documents
        st.session_state.titles = all_titles
        
        with st.expander("📚 Available Documents in RAG Knowledge base", expanded=False):
            for title in st.session_state.titles:
                st.write(f"- {title}")
    
        # Query input
        st.text("=== QUERY OPTIONS ===")
        st.text("▶️ Type your questions to query the knowledge base of your uploaded articles ")
        st.text("▶️ If you wish to download all images from the articles, type images ")
        st.text("▶️ If you want the system to find related articles to the ones you uploaded, type papers ")
        
        
        query = st.text_input(
            "Enter your Query:",
            placeholder="Query the knowledge base for response ...",
        )
    
        if query:
                
            if query == "papers".lower():
                st.text("🧠 Exporing my Knowledge Base ...")
                from search_articles_streamlit import execute_search_papers
                query_titles = all_titles
                
                for title in query_titles:
                    tar = title.split(" ")[:10]
                    col1, col2 = st.columns(2)
                    with col1:
                        st.text("🔎 " + " ".join(tar))
                    with col2:
                        execute_search_papers(" ".join(tar))
                
            elif query == "images".lower():
                all_bytes = list() ; all_ext = list()
                for file_path in session_dir.iterdir():
                    temp_bytes, temp_exts = read_pdf_image(file_path)
                    all_bytes.extend(temp_bytes)
                    all_ext.extend(temp_exts)
                
                # Crete download button for each image
                for i in range(len(all_ext)):
                    st.download_button(
                            label=f"Download Image_{i}",
                            data=bytes(all_bytes[i]),
                            file_name=f"image_{i}.{all_ext[i]}",
                            mime=f"image/{all_ext[i]}"
                        )
                st.text("All Images Processed Successfully! 😎")  
            
            else: 
                with st.spinner("Processing your query..."):
                    augmented_prompt = augment_prompt(
                        query, find_related_chunks(
                                        query, st.session_state.collection)
                        )
                    
                    response, references = rag_pipeline(
                        augmented_prompt, st.session_state.collection, st.session_state.llm_model
                    )    #query
    
                    # Display results in columns
                    col1, col2 = st.columns(2)
    
                    with col1:
                        st.markdown("### 🤖 Response")
                        st.write(response)
    
                    with col2:
                        st.markdown("### 📖 References Used")
                        for ref in references:
                            st.write(f"- {ref}")
    
                    # Show technical details in expander
                    with st.expander("🔍 Technical Details", expanded=False):
                        st.markdown("#### Augmented Prompt")
                        st.code(augmented_prompt)
    
                        st.markdown("#### Model Configuration")
                        st.write(f"- LLM Model: {llm_type.upper()}")
                        st.write(f"- Embedding Model: {embedding_type.upper()}")
    
    
        st.write("⚠️⚠️ Click the button below only after all your analysis. Clicking it deletes your uploaded files and closes your session. ⚠️⚠️")
        if st.button("End Analysis and Delete Files"):
            dest = move_to_permanent_storage(session_dir)
            st.success(f"Archived Files in {dest}... User session folder deleted permanently.")
            # Optional: reset session id so a fresh temp folder is made
            # if the user uploads more files in the same browser tab

            doc_ids = st.session_state.collection.get()["ids"]
            st.session_state.collection.delete(ids=doc_ids)
            #st.session_state.collection.delete()
            
            del st.session_state["session_id"]

    # No files uploaded. End active session
    except (UnboundLocalError, ValueError):
        st.write(f"{len(uploaded_files)} files uploaded. Session ended. Page will refresh to start a new session")
        del st.session_state["session_id"]
        streamlit_js_eval(js_expressions="parent.window.location.reload()")
        
if __name__ == "__main__":
    streamlit_app()
