import streamlit as st
from dotenv import load_dotenv
import pickle
import numpy as np
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import os

# Sidebar contents
with st.sidebar:
    st.image(os.path.join('assets', 'logo.png'), width = 100)
    add_vertical_space(2)
    st.image(os.path.join('assets', 'valeo_logo.png'), width = 200)
    add_vertical_space(1)
    st.title("Chatbot💬 - Contenus de Livres PDF")
    st.markdown('''
    ## Consignes
    En utilisant LLM, développer un chatbot qui interagit avec un livre pdf, pour répondre
    aux questions des utilisateurs, et les réponses sont basées sur le contenu du livre.
    ''')
    add_vertical_space(1)
    st.write('Conception: Mathieu RODRIGUES 🧑‍💻')

load_dotenv()

def create_faiss_index(embeddings):
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    embeddings_np = np.array(embeddings).astype('float32')
    index.add(embeddings_np)
    return index

def extract_full_sentence(response, context):
    """
    Extracts a full sentence containing the response from the context.
    """
    start_idx = response['start']
    end_idx = response['end']

    # Expand to the nearest sentence boundaries
    sentence_start = context.rfind('.', 0, start_idx) + 1
    sentence_end = context.find('.', end_idx)

    if sentence_end == -1:
        sentence_end = len(context)
    else:
        sentence_end += 1

    return context[sentence_start:sentence_end].strip()

def main():
    st.header("Chatbot 💬")

    embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')

    # upload a PDF file
    pdf = st.file_uploader("Téléverser (PDF)", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() if page.extract_text() else ''

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        with st.chat_message("human", avatar='🧑‍💻'):
            st.write(chunks[0]+ " (...)")

        # Embeddings
        store_name = pdf.name[:-4]
        #st.write(f'{store_name}')

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                index = pickle.load(f)
        else:
            embeddings = [embeddings_model.encode(text, show_progress_bar=True) for text in chunks]
            index = create_faiss_index(embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(index, f)

        # Accept user questions/query
        query = st.chat_input("Ask questions about your PDF file:")

        if query:
            with st.chat_message("human", avatar='🧑‍💻'):
                st.write(query)

            with st.chat_message("ai", avatar='🤖'):
                st.write("Let me some time to think...")
                loading_chat = st.empty()
                loading_chat.image(os.path.join('assets', 'loading.gif'))

            # Converting query to embedding
            query_embedding = embeddings_model.encode(query, show_progress_bar=False)
            query_embedding_np = np.array(query_embedding).reshape(1, -1).astype('float32')

            # Performing the search
            D, I = index.search(query_embedding_np, k=3)

            # Extracting the chunks corresponding to the indices
            docs = [chunks[i] for i in I[0]] 

            # BERT QA pipeline
            qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

            best_score = float('-inf')
            best_response = None
            full_sentence = ""
            best_context = ""

            # Iterate through the responses to find the one with the highest score
            for doc in docs:
                response = qa_pipeline(question=query, context=doc)
                if response['score'] > best_score:
                    best_score = response['score']
                    best_response = response['answer']
                    full_sentence = extract_full_sentence(response, doc)
                    best_context = doc

            loading_chat.empty() 
            with st.chat_message("assistant", avatar='🤖'):
                st.write("**Quick Answer:", best_response+"**")

            with st.chat_message("ai", avatar='🤖'):
                st.write("I'm now trying to generate an answer with a generative text model...")
                loading_chat2 = st.empty()
                loading_chat2.image(os.path.join('assets', 'loading.gif'))

            text_generator = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B")

            prompt = f"{query}\n\nContext:\n{best_context[:1000]}.\nKnowing that, i will answer to the question \"{query}\": "  # You can adjust the context length
            response = text_generator(prompt, max_length=50 + len(prompt), num_return_sequences=1)
            response = response[0]['generated_text'].split(f'Knowing that, i will answer to the question \"{query}\": ')[1]

            loading_chat2.empty()
            with st.chat_message("assistant", avatar='🤖'):
                st.write("Advanced Answer:", response)
                #st.write("Full Sentence:", full_sentence)

if __name__ == '__main__':
    main()
