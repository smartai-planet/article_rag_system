# Smart-Document-Analyzer
Smart Article: Document Analyzer with RAG Integration. Deployed and hosted on streamlit. Access through https://smart-article.streamlit.app  <br />
The fully deployed scripts can be found here: https://github.com/smartai-planet/article_rag_system

<img width="1611" height="487" alt="Screenshot 2026-07-22 022257" src="https://github.com/user-attachments/assets/5ed67b87-30f1-4171-ac7a-d5e3de048041" />

## How can Smart Article help you? Just Upload your document(s) and Smart Article will;

- Analyze your document(s) and embed your document(s) in a vector store through series of structured processes
- Provide semantic search & Maximal Marginal Relevance functionality through querying Smart Article, providing smart responses. Just ask whatever query, and you get a response!
- Responses are entirely based on the uploaded document(s), references for the responses are also provided
- Crawl the web and provide up to 20 other articles related to the document(s) uploaded, with download links. Just use the keyword: papers
- Extract all images from the document(s) uploaded, providing a download button. Just use the keyword: images
- Provide the opportunity to choose between different Model and Embedding Architectures

## WorkFlow for Deployed App
  
 - Quick House-cleaning, assign session ID to a user, create temporary session folder to hold files. Assigning sessions allows multiple users access the app simultaneously without conflict 
 - Select your choice of LLM and Embedding model.
 - Upload PDF file(s). Files are quickly parsed and contents are read
 - Contents are sliced into chunks, and each chunk assigned a chunk ID
 - Create vector store, with fixed embedding dimension of 768. This dimension allows using both Online and Local models (e.g Ollama llava)
 - User Enter their Query.
 - Query is converted to embedding and a similarity semantic search is done with the vector store to return similar chunks
 - Returned Similar chunks are wrapped with the query into a context knowledge, which is passed into a RAG pipeline.
 - The LLM in the RAG pipeline returns a smart and well structured response to the user based on their query, and references are also returned to enlighten users what part of the embedded context the response was derived from.
 - In the end, user session is closed and temporary session folder deleted.

## Python Libraries
- pymupdf==1.28.0
- pdfplumber==0.11.10
- openai==2.43.0
- chromadb==1.5.9
- pandas==2.3.3
- streamlit==1.59.2
- requests==2.34.2
