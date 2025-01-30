# run command in terminal 'pip install langchain langchain-ollama langchain-community pypdf docx2txt faiss-cpu'

import time

# requirements of RAG
from langchain_ollama.llms import OllamaLLM
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

start_time = time.time()

chat_llm = OllamaLLM(
    # ganti dengan model yang mau dipakai
    model="llama3.2"
    )

# ganti nama file 'sejarah.txt' dengan file yang spesifik
loader_text = TextLoader('sejarah.txt')
data_text = loader_text.load()

# pull model 'mxbai-embed-large' terlebih dahulu
embeddings = OllamaEmbeddings(
    model="mxbai-embed-large"
)

vector_store = FAISS.from_documents(data_text, embeddings)
chain = RetrievalQA.from_chain_type(llm=chat_llm, retriever=vector_store.as_retriever())

# Masukkan pertanyaan sesuai dengan isi file file (atau uji coba juga dengan pertanyaan diluar isi file tersebut)
query = "Kapan Universitas Muhammadiyah Surakarta di dirikan?"
print(f"Question: {query}")

result = chain.invoke(query)

# Menampilkan hasil
print(f"Answer: {result['result']}")

end_time = time.time()
total_time = end_time - start_time
print(f"Total Run-time: {total_time:.2f} seconds.")