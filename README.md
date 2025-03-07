# 📄 DocuMind AI - Your Intelligent Document Assistant  

## 📝 Introduction  
**DocuMind AI** is a **RAG (Retrieval-Augmented Generation) AI application** that allows users to upload PDFs and ask questions about their contents. It utilizes **LangChain** for document processing and retrieval, **Ollama** for running local LLMs, and **Mistral** for generating responses.  

---

## 🚀 Models Used  

| **Model** | **Usage** |
|-----------|----------|
| `mxbai-embed-large` | Converts document text into vector embeddings |
| `mistral` | Generates responses based on retrieved document context |

---

## 🔄 Flow of the Application  

1. **User uploads a PDF document** 🡪 Stored in a local directory.  
2. **Text is extracted and split into chunks** 🡪 Using `RecursiveCharacterTextSplitter`.  
3. **Chunks are converted to vector embeddings** 🡪 Using `OllamaEmbeddings` (`mxbai-embed-large`).  
4. **Embeddings are stored in an in-memory vector store** (`InMemoryVectorStore`).  
5. **User asks a question** 🡪 The app retrieves **similar document chunks** using cosine similarity.  
6. **The most relevant context is passed to Mistral LLM** 🡪 Generates a well-structured answer.  
7. **The response is displayed in the chat UI**.  

---

## 📜 Summary Flow  

--# 📄 DocuMind AI  

DocuMind AI is a **Retrieval-Augmented Generation (RAG)** application that allows users to upload PDFs and ask questions about their contents. It leverages **Mistral LLM** for text generation and **mxbai-embed-large** for vector embeddings.  







---


## 🛠️ Installation & Running Locally  

Follow these steps to run **DocuMind AI** on your local machine:  

### **1️⃣ Install Ollama**  
Ollama is required to run local LLMs. Download and install it from the official site:  

🔗 [Ollama Official Website](https://ollama.com)  

### **2️⃣ Pull Required Models**  
DocuMind AI uses **Mistral** for text generation and **mxbai-embed-large** for embeddings. Pull them using:  

```sh
ollama pull mistral
ollama pull mxbai-embed-large
```

### **3️⃣ Clone the Repository**

```sh
git clone https://github.com/notebook16/Document-RAG-GEN-AI.git
cd Document-RAG-GEN-AI
```

### **4️⃣ Create a Virtual Environment**

```sh
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

### **5️⃣ Install Dependencies**

```sh
pip install -r requirements.txt
```

### **6️⃣ Run the Application**

```sh
streamlit run app.py
```

---

## 🎯 Features  

✅ Upload and analyze PDFs 📄  
✅ Retrieve the most relevant information 🔎  
✅ Generate answers using Mistral LLM 🤖  
✅ Fast and efficient vector search with in-memory storage 🚀  

---

## 📜 License  

This project is open-source and available under the **MIT License**.  

---


