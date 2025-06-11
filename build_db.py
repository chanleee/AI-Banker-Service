import os
import pickle
import numpy as np
from openai import OpenAI
import tiktoken
from typing import List
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

class EmbeddingModel:
    def __init__(self, model_name="text-embedding-3-small"):
        self.client = client
        self.model_name = model_name
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = 8191

    def get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(input=text, model=self.model_name)
        return response.data[0].embedding

    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        return np.array([self.get_embedding(text) for text in texts])

    def split_documents(self, documents: List[str]) -> List[str]:
        # maybe chunking is necessary to get great result.
        return documents

class VectorDB:
    def __init__(self, directory="documents", vector_file="embeddings.npy"):
        self.directory = directory
        self.vector_file = vector_file
        self.chunks_file = os.path.splitext(vector_file)[0] + "_chunks.pkl"
        self.embedding_model = EmbeddingModel()

        if not os.path.exists(self.directory):
            print(f"'{self.directory}' dir does not exist. make dir and create financial products info txt files in it.")
            return

        docs = self._read_text_files()
        print(f"[DB Build] '{self.directory}' dir - {len(docs)} documents")
        
        self.chunks = self.embedding_model.split_documents(docs)
        print(f"[DB Build] {len(self.chunks)} separate n chunks.")
        
        with open(self.chunks_file, 'wb') as f:
            pickle.dump(self.chunks, f)
            print(f"[DB Build] chunks are saved - '{self.chunks_file}'")
        
        embeddings_data = self.embedding_model.get_embeddings_batch(self.chunks)
        self.embeddings = embeddings_data
        print(f"[DB Build] {len(self.embeddings)} embeddings are created.")
        
        self._store_embeddings()

    def _read_text_files(self):
        documents = []
        for filename in os.listdir(self.directory):
            if filename.endswith(".txt"):
                with open(os.path.join(self.directory, filename), 'r', encoding='utf-8') as file:
                    documents.append(file.read())
        return documents
    
    def _store_embeddings(self):
        np.save(self.vector_file, self.embeddings)
        print(f"[DB Build] embeddings are saved - '{self.vector_file}'")


# --- 메인 실행 ---
if __name__ == "__main__":
    print("create DB...")
    VectorDB(directory="documents", vector_file="financial_db.npy")
    print("\nComplete!")