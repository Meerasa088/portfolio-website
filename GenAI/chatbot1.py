import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

OPENAI_API_KEY = "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" #Pass your key here

st.header("Hi Meerasa, How Can i help you today.")

with st.sidebar:
    st.title("Your Documents")
    files = st.file_uploader(
        "Upload PDF files and start asking questions",
        type="pdf",
        accept_multiple_files=True
    )

# Extract the text from all uploaded PDFs
all_text = ""
if files:
    for file in files:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                all_text += page_text + "\n"

# Process if there’s any text
if all_text:
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(all_text)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    vector_store = FAISS.from_texts(chunks, embeddings)

    user_question = st.text_input("Ask anything here")

    if user_question:
        match = vector_store.similarity_search(user_question)

        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0,
            max_tokens=1000,
            model_name="gpt-3.5-turbo"
        )

        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=match, question=user_question)
        st.write(response)
else:
    st.info(" Please upload one or more PDF files to start.")


