import streamlit as st
from dotenv import load_dotenv
import os
from utils import *
import uuid

load_dotenv()  # ✅ Load .env at the very start

if 'unique_id' not in st.session_state:
    st.session_state['unique_id'] = ''


def main():
    st.set_page_config(page_title="Resume Screening Assistance")
    st.title("HR - Resume Screening Assistance...💁 ")
    st.subheader("I can help you in resume screening process")

    job_description = st.text_area("Please paste the 'JOB DESCRIPTION' here...", key="1")
    document_count = st.text_input("No.of 'RESUMES' to return", key="2")
    pdf = st.file_uploader("Upload resumes here, only PDF files allowed", type=["pdf"], accept_multiple_files=True)

    submit = st.button("Help me with the analysis")

    if submit:
        if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
            st.error(
                "ERROR: HUGGINGFACEHUB_API_TOKEN not found. Please set it in your .env file or Hugging Face Space Secrets.")
            return

        with st.spinner('Wait for it...'):
            st.session_state['unique_id'] = uuid.uuid4().hex
            final_docs_list = create_docs(pdf, st.session_state['unique_id'])

            st.write("*Resumes uploaded* :" + str(len(final_docs_list)))

            embeddings = create_embeddings_load_data()

            # --- CHROMADB REPLACEMENT ---
            push_to_chroma(st.session_state['unique_id'], embeddings, final_docs_list)
            relavant_docs = similar_docs(job_description, document_count, st.session_state['unique_id'])
            # ---------------------------

            st.write(":heavy_minus_sign:" * 30)

            for item in range(len(relavant_docs)):
                st.subheader("👉 " + str(item + 1))
                st.write("**File** : " + relavant_docs[item][0].metadata['name'])

                with st.expander('Show me 👀'):
                    st.info("**Match Score** : " + str(relavant_docs[item][1]))
                    summary = get_summary(relavant_docs[item][0])
                    st.write("**Summary** : " + summary)

        st.success("Hope I was able to save your time❤️")


if __name__ == '__main__':
    main()