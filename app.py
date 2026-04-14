import streamlit as st
import os
from rag.pipeline import build_rag_pipeline

st.set_page_config(page_title="RAG Assistant")

st.title("📄 RAG Document Assistant")
st.write("Upload a PDF and ask questions")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    if "qa_chain" not in st.session_state:
        st.info("Building AI pipeline...")
        st.session_state.qa_chain = build_rag_pipeline("temp.pdf")

    query = st.text_input("Ask your question:")

    if query:
        result = st.session_state.qa_chain.run(query)
        st.write("### 🤖 Answer:")
        st.write(result)