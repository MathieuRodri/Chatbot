import os
import pickle
import numpy as np
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

# Chargement des variables d'environnement
load_dotenv()

def create_faiss_index(embeddings):
    """
    Crée et retourne un index FAISS à partir des embeddings donnés.

    Args:
    embeddings (List[np.array]): Liste des embeddings (vecteurs de caractéristiques).

    Returns:
    faiss.IndexFlatL2: Index FAISS pour la recherche de proximité.
    """
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    embeddings_np = np.array(embeddings).astype('float32')
    index.add(embeddings_np)
    return index

def extract_full_sentence(response, context):
    """
    Extrait une phrase complète contenant la réponse du contexte.

    Args:
    response (dict): Réponse du modèle de question-réponse.
    context (str): Texte d'où la réponse a été extraite.

    Returns:
    str: Phrase complète contenant la réponse.
    """
    start_idx = response['start']
    end_idx = response['end']

    # Expansion aux limites les plus proches de la phrase
    sentence_start = context.rfind('.', 0, start_idx) + 1
    sentence_end = context.find('.', end_idx)
    sentence_end = sentence_end + 1 if sentence_end != -1 else len(context)

    return context[sentence_start:sentence_end].strip()

def main():
    # Configuration et contenu de la barre latérale
    with st.sidebar:
        st.image(os.path.join('assets', 'logo.png'))
        add_vertical_space(1)
        st.title("Chatbot💬 - Contenus de Livres PDF")
        st.markdown('''
        ## Consignes
        En utilisant LLM, développer un chatbot qui interagit avec un livre pdf, pour répondre
        aux questions des utilisateurs, et les réponses sont basées sur le contenu du livre.
        ''')
        add_vertical_space(1)
        st.write('Conception: Mathieu RODRIGUES 🧑‍💻')

    st.header("Chatbot (english only)💬")

    # Modèle pour les embeddings
    embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Téléversement d'un fichier PDF
    pdf = st.file_uploader("Téléverser (PDF)", type='pdf')

    if pdf is not None:
        # Lecture et extraction du texte du PDF
        pdf_reader = PdfReader(pdf)
        text = "".join([page.extract_text() or '' for page in pdf_reader.pages])

        # Division du texte en morceaux pour le traitement
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        with st.chat_message("human", avatar='🧑‍💻'):
            st.write(chunks[0] + " (...)")

        # Gestion des embeddings et de l'index FAISS
        store_name = pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                index = pickle.load(f)
        else:
            embeddings = [embeddings_model.encode(text, show_progress_bar=True) for text in chunks]
            index = create_faiss_index(embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(index, f)

        # Interaction avec l'utilisateur pour les questions
        query = st.chat_input("Ask questions about your PDF file:")

        if query:
            with st.chat_message("human", avatar='🧑‍💻'):
                st.write(query)

            # Recherche et réponse au chat
            handle_chat_response(query, chunks, embeddings_model, index)

def handle_chat_response(query, chunks, embeddings_model, index):
    """
    Gère la réponse au chat en fonction de la requête de l'utilisateur.

    Args:
    query (str): Question de l'utilisateur.
    chunks (List[str]): Morceaux de texte du PDF.
    embeddings_model (SentenceTransformer): Modèle pour générer des embeddings.
    index (faiss.IndexFlatL2): Index FAISS pour la recherche de proximité.
    """
    with st.chat_message("ai", avatar='🤖'):
        st.write("Let me some time to think...")
        loading_chat = st.empty()
        loading_chat.image(os.path.join('assets', 'loading.gif'))

    # Conversion de la requête en embedding et recherche
    query_embedding = embeddings_model.encode(query, show_progress_bar=False)
    query_embedding_np = np.array(query_embedding).reshape(1, -1).astype('float32')
    D, I = index.search(query_embedding_np, k=3)
    docs = [chunks[i] for i in I[0]]

    # Pipeline de question-réponse BERT
    qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")
    best_score, best_response, full_sentence, best_context = find_best_response(docs, qa_pipeline, query)

    # Affichage de la réponse rapide
    loading_chat.empty()
    with st.chat_message("assistant", avatar='🤖'):
        st.write("**Quick Answer:", best_response + "**")

    # Génération d'une réponse avancée
    generate_advanced_response(query, best_context)

def find_best_response(docs, qa_pipeline, query):
    """
    Trouve la meilleure réponse parmi les documents donnés en utilisant la pipeline QA.

    Args:
    docs (List[str]): Documents parmi lesquels chercher la réponse.
    qa_pipeline (transformers.Pipeline): Pipeline de question-réponse.
    query (str): Question posée.

    Returns:
    Tuple[float, str, str, str]: Meilleur score, meilleure réponse, phrase complète, meilleur contexte.
    """
    best_score = float('-inf')
    best_response = None
    full_sentence = ""
    best_context = ""

    for doc in docs:
        response = qa_pipeline(question=query, context=doc)
        if response['score'] > best_score:
            best_score = response['score']
            best_response = response['answer']
            full_sentence = extract_full_sentence(response, doc)
            best_context = doc

    return best_score, best_response, full_sentence, best_context

def generate_advanced_response(query, best_context):
    """
    Génère une réponse avancée en utilisant un modèle de génération de texte.

    Args:
    query (str): Question posée.
    best_context (str): Meilleur contexte trouvé pour la réponse.
    """
    with st.chat_message("ai", avatar='🤖'):
        st.write("I'm now trying to generate an answer with a generative text model...")
        loading_chat2 = st.empty()
        loading_chat2.image(os.path.join('assets', 'loading.gif'))

    text_generator = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B")
    prompt = f"{query}\n\nContext:\n{best_context[:1000]}.\nKnowing that, I will answer to the question \"{query}\": "  # Longueur du contexte ajustable
    response = text_generator(prompt, max_length=50 + len(prompt), num_return_sequences=1)
    response = response[0]['generated_text'].split(f'Knowing that, I will answer to the question \"{query}\": ')[1]

    loading_chat2.empty()
    with st.chat_message("assistant", avatar='🤖'):
        st.write("Advanced Answer:", response)

if __name__ == '__main__':
    main()
