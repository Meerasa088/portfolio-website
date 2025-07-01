
#streamlit---->it is used for creating user interface
#pypdf2 ----> to read the text in pdf we need this library
#langchain ---> It is responsible for convert user text to embeddings.
#faiss ----> Is responsible for store embeddings to vector database and also do similarity search(means if user asked question like what is the capital of india so in the
#vector store faiss search the similar  question and generate answer)

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

OPENAI_API_KEY="" # pass your API key here

# uploading the pdf files

st.header("This is my first Generative AI Chatbot ")

with st.sidebar:
    st.title("Generative AI pdf files")
    file=st.file_uploader("upload your pdf files and ask questions:",type="pdf")

#Extract pdf files

if file is not None:
    pdf_reader=PdfReader(file)
    text=""
    for page in pdf_reader.pages:
        text+=page.extract_text()
        #st.write(text)

    # Break  into the chunks
    text_splitter=RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=500,
        chunk_overlap=100,
        length_function=len

    )
    chunks=text_splitter.split_text(text)
    #st.write(chunk)

    # Generating embeddings
    embeddings=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    #create vector store
    vector_store=FAISS.from_texts(chunks,embeddings)

    # User questions
    user_question=st.text_input("Ask your questions to chatbot")

    #do similarity search
    if user_question:
        match=vector_store.similarity_search(user_question)
        #st.write(match)

        #define llm
        llm=ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0,
            max_tokens=500,
            model_name="gpt-3.5-turbo"
        )
        # generative output results
        #chain ---> take the user question , get the relevant document ,pass it to the llm , generate output
        chain=load_qa_chain(llm,chain_type="stuff")
        response=chain.run(input_documents=match,question=user_question)
        st.write(response)
