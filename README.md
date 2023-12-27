# Chatbot Development with PDF Book Contents


![ChatBot Logo](https://github.com/MathieuRodri/Chatbot/blob/main/assets/logo.png)


## Overview
This GitHub repository contains a Streamlit-based chatbot that interacts with PDF books to answer user queries using the contents of the book.

## Features
- **PDF Processing**: Converts and processes text from PDFs for AI model compatibility.
- **Text Splitting**: Implements techniques to split the book into manageable chunks.
- **Embeddings and Indexing**: Utilizes SentenceTransformer for embeddings and FAISS for efficient indexing.
- **Interactive Chatbot**: A user-friendly Streamlit interface for interacting with the chatbot.
- **Query Handling**: Employs BERT and GPT-Neo models for accurate and context-aware responses.

## Requirement
- Python 3.10.11 (not tested on others versions).

## Installation
1. [Download the stable version](https://github.com/MathieuRodri/Chatbot/releases/tag/1.1) or clone the repository.
2. To start the chatbot, run the `ChatBot.bat` script.

## Usage
- Run the Streamlit app (`ChatBot.bat`).
- Upload a PDF and interact with the chatbot to get responses based on the PDF content.

## Dependencies
- langchain==0.0.352
- langchain-community==0.0.6
- langchain-core==0.1.3
- PyPDF2==3.0.1
- python-dotenv==1.0.0
- streamlit==1.29.0
- streamlit-camera-input-live==0.2.0
- streamlit-card==1.0.0
- streamlit-embedcode==0.1.2
- streamlit-extras==0.3.6
- streamlit-faker==0.0.3
- streamlit-image-coordinates==0.1.6
- streamlit-keyup==0.2.2
- streamlit-toggle-switch==1.0.2
- streamlit-vertical-slider==2.5.5
- faiss-cpu==1.7.4
- streamlit-extras==0.3.6
- sentence-transformers==2.2.2
- sentence-transformers==2.2.2
- transformers==4.36.2
- accelerate==0.25.0


## Contribution
Developed by Mathieu RODRIGUES, a passionate student specializing in Computer Vision and Machine Learning.

## Contact
GitHub: [MathieuRodri](https://github.com/MathieuRodri)
