# Creator: Ikenna Oluigbo
# Contact me on callme-smartai@proton.me 
# Program to analyze document, summarize, and provide chat responses. ALSO, Project to suggest related articles online

#Importing Libraries 
import fitz
import pdfplumber
from openai import OpenAI
import os
import chromadb
import streamlit as st 
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction, OllamaEmbeddingFunction
import time


OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

pdfupload_path = "./pdf_files/"

class LLMModel:
    def __init__(self, model_type = "openai"): 
        self.model_type = model_type 
        if model_type == "openai":
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.model_name = "gpt-5.5"
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
                model_name="text-embedding-3-small"
            )
            
        elif model_type == "ollama_local":
            self.embedding_fn = OllamaEmbeddingFunction(
                url="http://localhost:11434", 
                model_name="qwen3-embedding"
            )
        
        elif model_type == "ollama_openai":
            self.embedding_fn = OpenAIEmbeddingFunction(
                api_key="ollama",
                api_base="http://localhost:11434/v1",
                model_name="nomic-embed-text"
            )    
            

# Function to split text
def make_chunks(texts : list, chunk_size: int = 1000, chunk_overlap: int = 200):
    chunks = []
    start = 0
    while start < len(texts):
        end = start + chunk_size
        chunks.append(texts[start:end])
        start = end - chunk_overlap
    
    return chunks


# Function to parse pdf texts
def read_pdfuploaded_text():
    all_documents = []
    for pdf in os.listdir(pdfupload_path):
        doc = fitz.open(pdfupload_path + pdf)
        print(f"Total lenght of Document Pages: {len(doc)} ")
        for i in range(len(doc)):
            curr_page = doc[i]
            text = [curr_page.get_text()]
            all_documents.extend(text)
            
    all_documents = " ".join(all_documents)         #Merge all strings into one
    
    return all_documents


#Extract Title
def read_pdf_title():
    all_titles = []
    #Method 1: Extract from Metadata
    for pdf in os.listdir(pdfupload_path):
        with pdfplumber.open(pdfupload_path + pdf) as p:
            page = p.pages[0]
            # Filter characters with a size greater than a threshold (e.g., 10)
            # Adjust threshold based on your PDF's font sizes
            filtered_page = page.filter(lambda x: x.get("size", 0) > 12)
            title = filtered_page.extract_text()
            all_titles.append(title)
            print(f"Heuristic Title: {title}")
    
    return all_titles


#Image Extraction        
def read_pdf_image():
    for pdf in os.listdir(pdfupload_path):
        doc = fitz.open(pdfupload_path + pdf)
        for page_num in range(len(doc)):   #Loop through all pages and find images from images (jpeg, jpg, png. OCR images cant be seen)
            page = doc[page_num]
            for img in page.get_images(full=True):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                with open(f"image_{page_num}.{image_ext}", "wb") as f:
                    f.write(image_bytes)
        
        
#Split Text and assign ID
def split_text_id():
    messages_with_id = dict()
    texts = read_pdfuploaded_text()
    chunks = make_chunks(texts=texts, chunk_size=1000, chunk_overlap=200)
    for i, chunk in enumerate(chunks):
        idx = i + 1
        messages_with_id[str(idx)] = chunk
    
    return messages_with_id


def setup_chromadb(texts, embedding_model):
    collection_name = "doc_analyzer"
    db_path = "./ChromaDB/llm_docdb"
    
    client = chromadb.PersistentClient(path=db_path)
  
    collection = client.get_or_create_collection(
        name=collection_name, 
        embedding_function=embedding_model.embedding_fn
    )
    
    
    ids = list(texts.keys())
    documents = list(texts.values())
    
    collection.add(ids=ids,
                      documents=documents)
   
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
        {"role": "system", "content": f"""You are a very smart and assistant.
                    You can answer questions about Subgraphs, Graphs embedding, Knowledge Graphs, Graph Neural Networks, Node translation, and lots more. 
                    You answers questions that are directly related to the sources/documents given, but can also be to add more knowledge to the source document. 
                    Use your base knowledge and the context information in {augmented_prompt} to answer questions"""},
        {"role": "user", "content": query}
        ]
    )
    
    print("\nGenerated response:")
    print(response)

    references = [chunk for chunk in relevant_chunks]
    return response, references


def streamlit_app():
    st.set_page_config(page_title="Smart Research Article Analyzer", layout="wide")
    st.title("🧾 Smart Research Article Analyzer with RAG Integration")

    # Sidebar for model selection
    st.sidebar.title("Model Configuration")

    llm_type = st.sidebar.radio(
        "Select LLM Model:",
        ["openai", "ollama"],
        format_func=lambda x: "OpenAI GPT (Most Stable)" if x == "openai" else "Ollama llava",
    )

    embedding_type = st.sidebar.radio(
        "Select Embedding Model:",
        ["openai", "ollama_local", "ollama_openai"],
        format_func=lambda x: {
            "openai": "OpenAI Embeddings (Most Stable)",
            "ollama_local": "Qwen3 Embedding (Ollama)",
            "ollama_openai": "Nomic Embed Text (Ollama)",
        }[x],
    )
    
    st.sidebar.image("./llm_bot.jpg")

    #Adding Footer text
    footer="""<style>
    a:link , a:visited{
    color: blue;
    background-color: transparent;
    text-decoration: underline;
    }
    
    a:hover,  a:active {
    color: red;
    background-color: transparent;
    text-decoration: underline;
    }
    
    .footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: white;
    color: black;
    text-align: center;
    }
    </style>
    <div class="footer">
    <p>Developed with ❤ by <a style='display: block; text-align: center;' target="_blank">Ikenna Oluigbo (PhD)</a></p>
    </div>
    """
    st.markdown(footer,unsafe_allow_html=True)
    

    st.write("#PDF Ingestion - RAG Knowledge Base ")

    if "pdf_docs" not in st.session_state:
        st.session_state.pdf_docs = []

    pdf_docs = st.file_uploader("""Upload ONLY PDF Files. You can upload one or more files (Upload within 60 seconds)""", 
                                    type="pdf", 
                                    accept_multiple_files=True,
                                    max_upload_size=5)    #MB

    if pdf_docs is not None:
        st.session_state.pdf_docs = pdf_docs
        
    if st.button("Submit & Process", disabled=len(st.session_state.pdf_docs) == 0):   
        for file in st.session_state.pdf_docs:
            st.write(f"Processing {file.name}...")
            # Create a temporary file to write the bytes to
            with open(f"./pdf_files/{file.name}", "wb") as temp_file:
                temp_file.write(file.read())
                
    time.sleep(60)

    st.write(f"All {len(st.session_state.pdf_docs)} files processed successfully! ✅")  


    # Initialize session state
    if "initialized" not in st.session_state:
        st.session_state.initialized = False    
        
        st.session_state.facts = split_text_id()

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
    st.session_state.titles = read_pdf_title()
    
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
            all_titles = read_pdf_title()
            
            for title in all_titles:
                tar = title.split(" ")[:5]
                col1, col2 = st.columns(2)
                with col1:
                    st.text("🔎 " + " ".join(tar))
                with col2:
                    execute_search_papers(" ".join(tar))
            
        elif query == "images".lower():
            read_pdf_image()
            st.text("All Images Downloaded Successfully! 😎")
        
        else: 
            with st.spinner("Processing your query..."):
                augmented_prompt = augment_prompt(
                    query, find_related_chunks(
                                    query, st.session_state.collection)
                    )
                
                response, references = rag_pipeline(
                    query, st.session_state.collection, st.session_state.llm_model
                )

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


if __name__ == "__main__":
    streamlit_app()
