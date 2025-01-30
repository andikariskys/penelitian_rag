import os
from flask import Flask, request, jsonify
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

app = Flask(__name__)

model = OllamaLLM(model="llama3.2:latest")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

@app.route('/')
def running():
    return "<p>Musai already running</p>"

@app.route('/', methods=['POST'])
def chat():
    try:
        message = request.json.get('message')
        if not message:
            return jsonify({"error": "Message wajib diisi!"}), 400
        return jsonify({"result": model.invoke(message)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def test():
    try: 
        message = request.form['message']
        if not message:
            return jsonify({"error": "Message wajib diisi!"}), 400
        return jsonify({"result": model.invoke(message)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload_file', methods=['POST'])
def upload_file():
    try:
        file = request.files['file']
        message = request.form['message']

        if not file:
            return jsonify({"error": "File wajib diupload!"}), 400

        if not message:
            return jsonify({"error": "Message wajib diisi!"}), 400

        filename, file_extension = os.path.splitext(file.filename)
        if file_extension != '.txt':
            return jsonify({"error": "Saat ini hanya mendukung file .txt saja"}), 400

        # Buat folder jika belum ada
        save_path = "documents"
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, file.filename)
        file.save(file_path)

        loader_text = TextLoader(file_path)
        data = loader_text.load()
        vector_store = FAISS.from_documents(data, embeddings)
        chain = RetrievalQA.from_chain_type(llm=model, retriever=vector_store.as_retriever())
        result = chain.invoke(message)

        return jsonify({"result": result['result']})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
