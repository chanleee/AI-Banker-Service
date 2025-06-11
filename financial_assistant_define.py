import os
import numpy as np
from playsound import playsound
import pickle

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

def get_relevant_context(query: str, k: int = 2):
    if not os.path.exists(DB_VECTORS_FILE) or not os.path.exists(DB_CHUNKS_FILE):
        return "DB does not exist. Please run 'build_db.py' first."

    stored_embeddings = np.load(DB_VECTORS_FILE)
    with open(DB_CHUNKS_FILE, 'rb') as f:
        chunks = pickle.load(f)
        
    query_embedding = EMBEDDING_MODEL.get_embedding(query)
    
    similarities = [cosine_similarity(query_embedding, emb) for emb in stored_embeddings]
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    
    top_k_chunks = [chunks[i] for i in top_k_indices]
    
    context = "\n\n---\n\n".join(top_k_chunks)
    print(f"[RAG] information search:\n{context[:300]}...")
    return context

def get_transcript(file_path: str) -> str:
    with open(file_path, "rb") as audio_file:
        print("[STT] convert audio to text...")
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
            language="ko"
        )
    print(f"[STT] text converted: {transcript}")
    return transcript

def get_ai_response(user_query: str):

    context = get_relevant_context(user_query)
    
    system_prompt = """
    당신은 친절하고 유능한 금융 도우미입니다.
    주어진 '참고 정보'를 바탕으로 사용자의 '질문'에 대해 명확하고 이해하기 쉽게 한국어로 답변해주세요.
    정보에 없는 내용은 추측해서 말하지 말고, 정보가 부족하다고 솔직하게 말해주세요.
    """
    
    human_prompt = f"""
    ### 참고 정보:
    {context}
    
    ### 질문:
    {user_query}
    
    ### 답변:
    """
    
    print("[LLM] AI generating its response...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_prompt}
        ],
        temperature=0.2
    )
    ai_answer = response.choices[0].message.content
    print(f"[LLM] answer generated: {ai_answer}")
    return ai_answer

def play_ai_response_with_tts(text_response: str):
    print("[TTS] converting answer to speech...")
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text_response
    )
    response.stream_to_file(SPEECH_FILE_PATH)
    
    print("[TTS] play speech...")
    playsound(SPEECH_FILE_PATH)
    os.remove(SPEECH_FILE_PATH)

def process_voice_query(file_path: str):
    user_transcript = get_transcript(file_path)
    ai_response = get_ai_response(user_transcript)
    play_ai_response_with_tts(ai_response)