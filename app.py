import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader  #for loading document
from langchain_text_splitters import RecursiveCharacterTextSplitter #for splitting large text into chunks
from langchain_core.vectorstores import InMemoryVectorStore #for storing vectors and also find similarity between user query vector and stored vector using cosine similairy
from langchain_ollama import OllamaEmbeddings #for vector embedding
from langchain_core.prompts import ChatPromptTemplate #creating prompt retrieve from the document
from langchain_ollama.llms import OllamaLLM #for communicating with model
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    .research-header {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        color: #00FFAA;
        padding: 10px;
    }
     .subtext {
        text-align: center;
        font-size: 18px;
        color: #A0A0A0;
        margin-bottom: 20px;
    }
    /* Chat Input Styling */
    .stChatInput input {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
    }
    
    /* User Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1E1E1E !important;
        border: 1px solid #3A3A3A !important;
        color: #E0E0E0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Assistant Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #2A2A2A !important;
        border: 1px solid #404040 !important;
        color: #F0F0F0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Avatar Styling */
    .stChatMessage .avatar {
        background-color: #00FFAA !important;
        color: #000000 !important;
    }
    
    /* Text Color Fix */
    .stChatMessage p, .stChatMessage div {
        color: #FFFFFF !important;
    }
    
    .stFileUploader {
        background-color: #1E1E1E;
        border: 1px solid #3A3A3A;
        border-radius: 5px;
        padding: 15px;
    }
    
    h1, h2, h3 {
        color: #00FFAA !important;
    }
    /* Footer */
    .footer {
        text-align: center;
        font-size: 14px;
        color: #A0A0A0;
        margin-top: 40px;
    }
    </style>
    """, unsafe_allow_html=True)
# UI Layout
st.markdown('<p class="research-header">📄 DocuMind AI</p>', unsafe_allow_html=True)
st.markdown('<p class="subtext">Your Intelligent Research Assistant – Powered by Ollama & LangChain</p>', unsafe_allow_html=True)

st.markdown("---")


PROMPT_TEMPLATE = """
You are a knowledgeable AI assistant helping with document-based queries.
Use the provided context to answer accurately. If unsure, say "I don't know."

Query: {user_query} 
Context: 
{document_context} 

Provide a clear to the point and well-structured answer:
"""


PDF_STORAGE_PATH = 'document_store/pdfs' #path of uploaded document
EMBEDDING_MODEL = OllamaEmbeddings(model="mxbai-embed-large") #deepseek is also doing the vector embeddig here
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL) # for storing vector embediing
LANGUAGE_MODEL = OllamaLLM(model="mistral") #llm model


#uploading and saving file in path
def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path , "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path    

#load the file or pdf document
def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path) 
    return document_loader.load()   

#chunking the document
def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap=200,
        add_start_index = True
    )
    return text_processor.split_documents(raw_documents)


#convert the text into embedding
def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

#find similar document in the vectorstore
def find_related_documents(query, k=3):
    return DOCUMENT_VECTOR_DB.similarity_search(query,k=k)


#generating response from model
def generate_answer(user_query , context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})



# UI Configuration


st.title("📘 DocuMind AI")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")

# File Upload Section
st.subheader("📤 Upload Research Document (PDF)")
uploaded_pdf = st.file_uploader(
    "Select a PDF document for analysis",
    type="pdf",
    help="Upload a research paper, report, or study document",
    accept_multiple_files=False
)


if uploaded_pdf:
    saved_path = save_uploaded_file(uploaded_pdf)  #save pdf in path
    raw_docs = load_pdf_documents(saved_path) #load the file from path
    processed_chunks = chunk_documents(raw_docs) #break the doc into chunks
    index_documents(processed_chunks)  #convert text chunks into vector embedding using deepseek and stor in vectore store
    
    st.success("✅ Document processed successfully! Ask your questions below.")
    
    user_input = st.chat_input("🔍 Ask a question about the document...")
    
    if user_input:
        with st.chat_message("user", avatar="🧑‍💻"):
            st.write(user_input)
        
        with st.spinner("🔎 Analyzing document..."):
            relevant_docs = find_related_documents(user_input)
            ai_response = generate_answer(user_input, relevant_docs)
            
        with st.chat_message("assistant", avatar="🤖"):
            st.write(ai_response)    

# Footer
st.markdown('<p class="footer">Built with ❤️ using Ollama, LangChain, Mistral 7B & MXBAI-Embed-Large</p>', unsafe_allow_html=True)            