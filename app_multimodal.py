# Importing Libraries
import re
import shutil
import tempfile
import time
import uuid
from io import BytesIO
from pathlib import Path
 
import chromadb
import fitz  # PyMuPDF
import pdfplumber
import requests
import streamlit as st
import trafilatura
from bs4 import BeautifulSoup
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_community.document_loaders import YoutubeLoader
from openai import OpenAI
from pypdf import PdfReader
 
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # fall back to the import you were using
    from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
 
 
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
 
# ---- Config ----
BASE_TMP_DIR = Path(tempfile.gettempdir()) / "myapp_sessions"
PERMANENT_DIR = Path("permanent_uploads")  # where finished files end up
MAX_SESSION_AGE_SECONDS = 0.5 * 60 * 60    # delete abandoned temp folders after 30 minutes
DB_PATH = "./ChromaDB/llm_docdb"
 
BASE_TMP_DIR.mkdir(parents=True, exist_ok=True)
PERMANENT_DIR.mkdir(parents=True, exist_ok=True)
 
EMBEDDING_INFO = {
    "openai": {"label": "OpenAI Embeddings (Most Stable)", "dimensions": 1536},
    "ollama_Nomic": {"label": "Nomic Embed Text (Ollama)", "dimensions": 768},
}
 
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
 
 
# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class LLMModel:
    def __init__(self, model_type="openai"):
        self.model_type = model_type
        if model_type == "openai":
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.model_name = "gpt-6-luna"
        else:
            self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            self.model_name = "llava"
 
    def generate_completion(self, messages):
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating content: {str(e)}"
 
    def stream_completion(self, messages):
        """Generator that yields the response text piece by piece."""
        try:
            stream = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"Error generating content: {str(e)}"
 
 
class EmbeddingModel:
    def __init__(self, model_type="openai"):
        self.model_type = model_type
        self.dimensions = EMBEDDING_INFO[model_type]["dimensions"]
        if model_type == "openai":
            self.embedding_fn = OpenAIEmbeddingFunction(
                api_key=OPENAI_API_KEY,
                model_name="text-embedding-3-small",
                dimensions=self.dimensions,
            )
        else:  # "ollama_Nomic"  (the old code used `==` here, which did nothing)
            self.embedding_fn = OpenAIEmbeddingFunction(
                api_key="ollama",
                api_base="http://localhost:11434/v1",
                model_name="nomic-embed-text",
            )
 
 
# --------------------------------------------------------------------------
# Session / file handling
# --------------------------------------------------------------------------
def cleanup_stale_sessions(max_age_seconds: float = MAX_SESSION_AGE_SECONDS):
    """Remove session folders older than max_age_seconds."""
    now = time.time()
    for folder in BASE_TMP_DIR.iterdir():
        if folder.is_dir() and now - folder.stat().st_mtime > max_age_seconds:
            shutil.rmtree(folder, ignore_errors=True)
 
 
def get_session_dir() -> Path:
    """Create (once per session) a unique temp folder for this user."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
 
    session_dir = BASE_TMP_DIR / st.session_state.session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir
 
 
def move_to_permanent_storage(session_dir: Path) -> Path:
    """Move all files from the session temp folder to permanent storage,
    then delete the temp folder."""
    dest = PERMANENT_DIR / st.session_state.session_id
    dest.mkdir(parents=True, exist_ok=True)
 
    for file_path in session_dir.iterdir():
        shutil.move(str(file_path), str(dest / file_path.name))
 
    shutil.rmtree(session_dir, ignore_errors=True)
    return dest
 
 
# --------------------------------------------------------------------------
# Text extraction
# --------------------------------------------------------------------------
def clean_title(title, fallback: str) -> str:
    title = " ".join((title or "").split())
    return title[:150] if title else fallback
 
 
def read_pdfuploaded_text(pdf_path) -> str:
    parts = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            parts.append(page.get_text())
    return "\n".join(parts)
 
 
def read_pdf_title(pdf_path):
    """Heuristic title: text on page 1 with a large font size."""
    try:
        with pdfplumber.open(pdf_path) as p:
            page = p.pages[0]
            filtered_page = page.filter(lambda x: x.get("size", 0) > 12)
            return filtered_page.extract_text()
    except Exception as e:
        print(f"[title extraction failed] {pdf_path} -> {e}")
        return None
 
 
def read_pdf_image(pdf_path):
    temp_bytes, temp_ext = [], []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            for img in page.get_images(full=True):
                base_image = doc.extract_image(img[0])
                temp_bytes.append(base_image["image"])
                temp_ext.append(base_image["ext"])
    return temp_bytes, temp_ext
 
 
def extract_pdf_text(content: bytes) -> str:
    reader = PdfReader(BytesIO(content))
    return "\n".join(page.extract_text() or "" for page in reader.pages)
 
 
def extract_html_text(content: bytes) -> str:
    try:
        text = trafilatura.extract(content)
        if text and text.strip():
            return text
    except Exception as e:
        print(f"[trafilatura failed, falling back] {e}")
 
    soup = BeautifulSoup(content, "html.parser")
    return soup.get_text(separator="\n", strip=True)
 
 
def process_link(url: str) -> str | None:
    """Download a web page or PDF link and return its text (or None)."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[Request failed] {url} -> {e}")
        return None
 
    content_type = response.headers.get("Content-Type", "")
 
    try:
        if "pdf" in content_type.lower() or url.lower().split("?")[0].endswith(".pdf"):
            text = extract_pdf_text(response.content)
        else:
            text = extract_html_text(response.content)
    except Exception as e:
        print(f"[Extraction failed] {url} -> {e}")
        return None
 
    if not text or not text.strip():
        print(f"[No text extracted] {url}")
        return None
 
    return text
 
 
def load_youtube_text(url: str) -> str:
    """Return the transcript text, or "" if unavailable."""
    docs = YoutubeLoader.from_youtube_url(url, add_video_info=False).load()
    return docs[0].page_content if docs else ""
 
 
# --------------------------------------------------------------------------
# Chunking  (one function for every source type)
# --------------------------------------------------------------------------
def make_chunks(text: str, source: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Split text into chunks. Returns a list of {"id", "text", "metadata"} dicts
    with plain-string text and scalar metadata, which is what Chroma needs."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return [
        {"id": str(uuid.uuid4()), "text": piece, "metadata": {"source": source}}
        for piece in splitter.split_text(text)
        if piece.strip()
    ]
 
 
# --------------------------------------------------------------------------
# Vector store
# --------------------------------------------------------------------------
def build_collection(chunks, embedding_model, embedding_type):
    """(Re)build this session's Chroma collection from the given chunks."""
    client = chromadb.PersistentClient(path=DB_PATH)
 
    # Drop the previous collection for this session (e.g. embedding model changed)
    old_name = st.session_state.get("collection_name")
    if old_name:
        try:
            client.delete_collection(old_name)
        except Exception:
            pass
 
    # One collection per session AND embedding model (dimensions differ: 1536 vs 768)
    name = f"doc_{st.session_state.session_id}_{embedding_type}"
    try:
        client.delete_collection(name)
    except Exception:
        pass
 
    collection = client.create_collection(
        name=name, embedding_function=embedding_model.embedding_fn
    )
 
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )
 
    st.session_state.collection_name = name
    return collection
 
 
def delete_collection():
    name = st.session_state.get("collection_name")
    if name:
        try:
            chromadb.PersistentClient(path=DB_PATH).delete_collection(name)
        except Exception:
            pass
 
 
def find_related_chunks(query, collection, top_k=5):
    results = collection.query(query_texts=[query], n_results=top_k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return docs, metas
 
 
def augment_prompt(query, relevant_chunks):
    context = "\n".join(relevant_chunks)
    return f"Context:\n{context}\n\nQuestion: {query}"
 
 
SYSTEM_PROMPT = """Your name is SMART. You are a very smart, knowledgeable assistant.
 
Answer the user's question using ONLY the context information provided from the documents
they uploaded in this session. Do not use information from any other user's session.
 
If you are not sure, ask the user for keywords from their uploaded documents so you can
narrow down the context.
 
If the answer is not in the context, you may use your base knowledge, but you must tell the
user clearly that the answer was not found in the context and that you are drawing on your
general knowledge.
 
### Polite Response:
- Acknowledge the user's query and express gratitude for the opportunity to assist.
- Give a clear, concise answer that directly addresses the question, grounded in the context.
  Avoid ambiguity and hallucination.
- Use positive language and a supportive tone.
- Include extra resources only when necessary.
- Conclude by inviting follow-up questions."""
 
 
def rag_pipeline(query, collection, llm_model, top_k=5):
    """Takes the RAW user query (it is augmented here, once).
    Retrieval happens immediately; the returned `stream` is a generator
    that yields the LLM's answer as it is produced."""
    relevant_chunks, metadatas = find_related_chunks(query, collection, top_k)
    prompt = augment_prompt(query, relevant_chunks)
 
    stream = llm_model.stream_completion(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )
 
    references = list(zip(relevant_chunks, metadatas))
    return stream, references, prompt
 
 
# --------------------------------------------------------------------------
# Streamlit app
# --------------------------------------------------------------------------
def streamlit_app():
    cleanup_stale_sessions()
 
    st.set_page_config(page_title="Smart Research Article Analyzer", layout="wide")
    st.title("🧾 Smart Article: Document Analyzer with RAG Integration")
 
    st.caption("How can Smart Article help you? Just upload your document(s) and Smart Article will:")
    st.badge("- Analyze your document(s) and embed them in a vector store through a series of structured processes")
    st.badge("- Provide semantic search through querying Smart Article, providing smart responses. Just ask whatever query, and you get a response!")
    st.badge("- Responses are entirely based on the uploaded document(s); references for the responses are also provided")
    st.badge("- Crawl the web and provide up to 20 other articles related to the document(s) uploaded, with download links. Just use the keyword: papers")
    st.badge("- Extract all images from the document(s) uploaded, providing a download button. Just use the keyword: images")
    st.badge("- Provide the opportunity to choose between different Model and Embedding Architectures")
 
    # Message left over from the previous run (e.g. after ending a session)
    flash = st.session_state.pop("flash", None)
    if flash:
        st.success(flash)
 
    # ---- Sidebar ----
    st.sidebar.title("Model Configuration")
 
    llm_type = st.sidebar.radio(
        "Select LLM Model:",
        ["openai", "ollama"],
        format_func=lambda x: "OpenAI GPT (Most Stable)" if x == "openai" else "Ollama llava",
    )
 
    embedding_type = st.sidebar.radio(
        "Select Embedding Model:",
        list(EMBEDDING_INFO.keys()),
        format_func=lambda x: EMBEDDING_INFO[x]["label"],
    )
 
    emb_name = " ".join(embedding_type.split("_")).upper()
    st.sidebar.write(
        "Current Embedding Model:\n\n"
        f"- Name: {emb_name}\n"
        f"- Dimensions: {EMBEDDING_INFO[embedding_type]['dimensions']}"
    )
 
    if Path("./llm_bot.jpg").exists():
        st.sidebar.image("./llm_bot.jpg")
 
    # ---- Footer ----
    footer = """
    <style>
    a:link, a:visited { color: blue; text-decoration: underline; }
    a:hover, a:active { color: red; text-decoration: underline; }
    .footer {
        position: fixed; left: 0; bottom: 0; width: 100%;
        background: white; color: black; text-align: center;
        padding: 10px; z-index: 9999;
    }
    </style>
    <div class="footer">
        <p>
            Developed for: 
            <a href="https://github.com/ikenna-oluigbo/" target="_blank">Smart Article 2026.</a>
            Drop a feedback!
        </p>
    </div>
    """
    st.markdown(footer, unsafe_allow_html=True)
 
    # ---- Session state ----
    session_dir = get_session_dir()
    st.caption(f"Session ID: {st.session_state.session_id}")
 
    # sources: {key: {"type": "pdf"|"web"|"youtube", "title": str, "chunks": [...]}}
    # key = file name for PDFs, the URL for web/YouTube sources.
    if "sources" not in st.session_state:
        st.session_state.sources = {}
    if "done_uploading" not in st.session_state:
        st.session_state.done_uploading = False
 
    sources = st.session_state.sources
 
    st.write("RAG KNOWLEDGE BASE - Upload a PDF file, provide a Web URL or provide a YouTube URL")
 
    tab1, tab2, tab3 = st.tabs(["📁 Upload PDF File", "🔗 From Web URL", "🔗 From YouTube URL"])
 
    # ---------------- Tab 1: PDFs ----------------
    with tab1:
        uploaded_files = st.file_uploader(
            "Upload files (you can upload in multiple batches). "
            "Click 'Done Uploading' after all your file selection.",
            accept_multiple_files=True,
            key="uploader",
            type="pdf",
            max_upload_size=10,
        )
        current_names = {f.name for f in (uploaded_files or [])}
 
        # Drop PDFs the user removed from the uploader
        for key, src in list(sources.items()):
            if src["type"] == "pdf" and key not in current_names:
                (session_dir / key).unlink(missing_ok=True)
                del sources[key]
 
        # Process each new PDF exactly once
        for f in uploaded_files or []:
            if f.name in sources:
                continue
            save_path = session_dir / f.name
            try:
                save_path.write_bytes(f.getbuffer())
                with st.spinner(f"Analyzing: {f.name}"):
                    text = read_pdfuploaded_text(save_path)
                    if not text.strip():
                        st.warning(f"No extractable text in {f.name} (scanned PDF?). Skipped.")
                        save_path.unlink(missing_ok=True)
                        continue
                    chunks = make_chunks(text, f.name, chunk_size=1000, chunk_overlap=200)
                    title = clean_title(read_pdf_title(save_path), f.name)
                sources[f.name] = {"type": "pdf", "title": title, "chunks": chunks}
            except Exception as e:
                st.error(f"Could not process {f.name}: {e}")
                save_path.unlink(missing_ok=True)
 
        n_pdf = sum(1 for s in sources.values() if s["type"] == "pdf")
        st.write(f"PDF files ready: {n_pdf}")
 
    # ---------------- Tab 2: Web URL ----------------
    with tab2:
        weburl = st.text_input("Enter Web URL", placeholder="https://github.com/ikenna-oluigbo", key="weburl_input")
        if st.button("Add URL", key="add_web"):
            weburl = weburl.strip()
            if not weburl.startswith(("http://", "https://")):
                st.error("Please enter a full URL starting with http:// or https://")
            elif weburl in sources:
                st.info("That URL has already been added.")
            else:
                with st.spinner(f"Analyzing: {weburl}"):
                    text = process_link(weburl)
                if not text:
                    st.error("Couldn't extract any text from that URL.")
                else:
                    sources[weburl] = {
                        "type": "web",
                        "title": clean_title(weburl, weburl),
                        "chunks": make_chunks(text, weburl, chunk_size=500, chunk_overlap=150),
                    }
                    st.success("URL added.")
 
        web_links = [k for k, s in sources.items() if s["type"] == "web"]
        st.write(f"{len(web_links)} web link(s) added")
        for link in web_links:
            st.write(f"- {link}")
 
    # ---------------- Tab 3: YouTube ----------------
    with tab3:
        yturl = st.text_input("Enter YouTube URL", placeholder="https://www.youtube.com/watch?v=...", key="yt_input")
        if st.button("Add video", key="add_yt"):
            yturl = yturl.strip()
            if not yturl:
                st.error("Please enter a YouTube URL.")
            elif yturl in sources:
                st.info("That video has already been added.")
            else:
                try:
                    with st.spinner(f"Loading transcript: {yturl}"):
                        text = load_youtube_text(yturl)
                except Exception as e:
                    st.error(f"Could not load the transcript: {e}")
                    text = ""
                if text.strip():
                    sources[yturl] = {
                        "type": "youtube",
                        "title": clean_title(yturl, yturl),
                        "chunks": make_chunks(text, yturl, chunk_size=500, chunk_overlap=150),
                    }
                    st.success("Video added.")
                elif text == "":
                    st.warning("No transcript found for that video.")
 
        yt_links = [k for k, s in sources.items() if s["type"] == "youtube"]
        st.write(f"{len(yt_links)} video(s) added")
        for link in yt_links:
            st.write(f"- {link}")
 
    # ---------------- Done Uploading ----------------
    if st.button("Done Uploading"):
        if not sources:
            st.warning("Add at least one PDF, web link or YouTube video first.")
        else:
            st.session_state.done_uploading = True
 
    if not (st.session_state.done_uploading and sources):
        return
 
    st.write("All processing completed successfully")
 
    # (Re)build the vector index whenever the sources or embedding model change
    signature = (tuple(sorted(sources)), embedding_type)
    if st.session_state.get("index_signature") != signature:
        all_chunks = [c for s in sources.values() for c in s["chunks"]]
        with st.spinner("Embedding your documents..."):
            try:
                st.session_state.collection = build_collection(
                    all_chunks, EmbeddingModel(embedding_type), embedding_type
                )
                st.session_state.index_signature = signature
            except Exception as e:
                st.error(f"Failed to build the vector store: {e}")
                return
 
    llm_model = LLMModel(llm_type)
    all_titles = [s["title"] for s in sources.values()]
 
    with st.expander("📚 Available Documents in RAG Knowledge base", expanded=False):
        for title in all_titles:
            st.write(f"- {title}")
 
    # ---- Query input ----
    st.text("=== QUERY OPTIONS ===")
    st.text("▶️ Type your questions to query the knowledge base of your uploaded articles ")
    st.text("▶️ If you wish to download all images from the PDFs, type: images ")
    st.text("▶️ If you want the system to find related articles to the ones you uploaded, type: papers ")
 
    query = st.text_input("Enter your Query:", placeholder="Query the knowledge base for response ...")
    query = query.strip()
 
    if query:
        if query.lower() == "papers":
            st.text("🧠 Exploring my Knowledge Base ...")
            from search_articles_streamlit import execute_search_papers
 
            for title in all_titles:
                tar = " ".join(title.split(" ")[:10])
                col1, col2 = st.columns(2)
                with col1:
                    st.text("🔎 " + tar)
                with col2:
                    execute_search_papers(tar)
 
        elif query.lower() == "images":
            all_bytes, all_ext = [], []
            for file_path in sorted(session_dir.glob("*.pdf")):
                temp_bytes, temp_exts = read_pdf_image(file_path)
                all_bytes.extend(temp_bytes)
                all_ext.extend(temp_exts)
 
            if not all_bytes:
                st.info("No extractable images found (images only come from uploaded PDFs).")
            for i, (img_bytes, ext) in enumerate(zip(all_bytes, all_ext)):
                st.download_button(
                    label=f"Download Image_{i}",
                    data=bytes(img_bytes),
                    file_name=f"image_{i}.{ext}",
                    mime=f"image/{ext}",
                    key=f"img_dl_{i}",
                )
            st.text("All Images Processed Successfully! 😎")
 
        else:
            with st.spinner("Searching your documents..."):
                stream, references, prompt = rag_pipeline(
                    query, st.session_state.collection, llm_model
                )
 
            col1, col2 = st.columns(2)
 
            # Fill the references column first so it is visible while the answer streams
            with col2:
                st.markdown("### 📖 References Used")
                for doc, meta in references:
                    st.write(f"- **{meta.get('source', 'unknown')}**: {doc}")
 
            with col1:
                st.markdown("### 🤖 Response")
                st.write_stream(stream)
 
            with st.expander("🔍 Technical Details", expanded=False):
                st.markdown("#### Augmented Prompt")
                st.code(prompt)
 
                st.markdown("#### Model Configuration")
                st.write(f"- LLM Model: {llm_type.upper()}")
                st.write(f"- Embedding Model: {embedding_type.upper()}")
 
    # ---- End session ----
    st.write("⚠️⚠️ Click the button below only after all your analysis. It closes your session. ⚠️⚠️")
    if st.button("End Analysis and Delete Files"):
        delete_collection()
        dest = move_to_permanent_storage(session_dir)
        flash_msg = f"Archived files in {dest}. Session closed."
 
        # Reset everything so the next upload starts a fresh session
        for key in [
            "session_id", "sources", "done_uploading", "collection",
            "collection_name", "index_signature", "uploader",
            "weburl_input", "yt_input",
        ]:
            st.session_state.pop(key, None)
        st.session_state["flash"] = flash_msg
        st.rerun()
 
 
if __name__ == "__main__":
    streamlit_app()
