import faiss
import openai
import os
import pygame
import speech_recognition as sr
from gtts import gTTS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Set OpenAI API Key
openai.api_key = "[your api key ]"

# Function to load and split a PDF
def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # Split text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    
    return [chunk.page_content for chunk in chunks]

# Function to create FAISS vector database
def create_faiss_index(chunks):
    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.from_texts(chunks, embeddings)
    return vector_db

# Function to capture speech input
def listen_to_question():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n ask your question...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        question = recognizer.recognize_google(audio)
        print(f"🗣️ You asked: {question}")
        return question
    except sr.UnknownValueError:
        print("❌ Could not understand the question. Try again.")
        return None
    except sr.RequestError:
        print("❌ Speech recognition service is unavailable.")
        return None

# Function to generate and speak the answer
def speak_answer(answer):
    print(f"🤖 Answer: {answer}")

    # Convert text to speech
    tts = gTTS(text=answer, lang="en", slow=False)
    audio_file = "answer.mp3"
    tts.save(audio_file)

    # Play the generated speech
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        continue

    pygame.mixer.quit()
    os.remove(audio_file)

# Function to answer questions using RAG
def ask_question(vector_db, question):
    embeddings = OpenAIEmbeddings()
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 relevant chunks
    
    # Get relevant docs
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Use OpenAI to generate an answer
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant answering questions based on provided context."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ]
    )
    
    return response["choices"][0]["message"]["content"]

# Main execution
if __name__ == "__main__":
    pdf_path = input("Enter the path of the PDF file: ")
    
    if not os.path.exists(pdf_path):
        print("Error: PDF file not found!")
        exit(1)
    
    print("📄 Loading and processing the PDF...")
    chunks = load_and_split_pdf(pdf_path)
    
    print("⚡ Creating FAISS index...")
    vector_db = create_faiss_index(chunks)
    
    print("\n✅ Ready! Speak your question.")
    while True:
        question = listen_to_question()
        if question:
            answer = ask_question(vector_db, question)
            speak_answer(answer)
