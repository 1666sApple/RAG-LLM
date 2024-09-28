# ChatGroq Demo

This project demonstrates a Streamlit-based web application that uses the ChatGroq language model to answer questions based on the content of a specified website. The application utilizes LangChain for document processing, embeddings, and retrieval.

## Features

- Web scraping and document loading from a specified URL
- Text splitting and embedding generation
- Vector storage using FAISS for efficient similarity search
- Integration with the ChatGroq language model
- Real-time question answering based on the loaded content
- Response time measurement
- Expandable view of similar documents used for context

## Prerequisites

- Python 3.7+
- Groq API key

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/1666sApple/RAG-LLM
   cd RAG-LLM
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   Create a `.env` file in the project root and add your Groq API key:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Enter your question in the text input field and press Enter.

4. The app will display the generated response, response time, and an expandable section showing the similar documents used for context.

## How it Works

1. The application loads content from the specified URL (https://docs.smith.langchain.com/ in this case).
2. The loaded content is split into smaller chunks for processing.
3. These chunks are embedded using the HuggingFace embeddings model.
4. The embeddings are stored in a FAISS vector store for efficient similarity search.
5. When a user enters a question, the app retrieves relevant document chunks from the vector store.
6. The retrieved chunks are used as context for the ChatGroq model to generate an answer.
7. The response, along with the response time and similar documents, is displayed to the user.

## Customization

- To change the source website, modify the URL in the `WebBaseLoader` initialization.
- To use a different embedding model, update the `model_name` in the `HuggingFaceEmbeddings` initialization.
- To adjust the chunk size for text splitting, modify the `chunk_size` and `chunk_overlap` parameters in the `RecursiveCharacterTextSplitter` initialization.

## Troubleshooting

If you encounter any issues with the Groq API key, ensure that it is correctly set in your `.env` file and that you have restarted the application after setting it.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.