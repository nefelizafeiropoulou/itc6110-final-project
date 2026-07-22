import streamlit as st
st.set_page_config(page_title="Colosseum Review Assistant")
st.title("Colosseum Review Assistant")
st.write("Ask a question about visitor reviews.")
question=st.text_input("Your question")
if st.button("Ask") and question:
    st.info("Connect the saved retriever and generator from the RAG cell here.")
    st.write("Question:",question)
    st.subheader("Answer")
    st.write("The answer should be generated from the retrieved review context.")
    with st.expander("Retrieved reviews"): st.write("Display the top-k retrieved reviews here.")
