import streamlit as st
import json

from sentence_transformers import SentenceTransformer, util
from streamlit_lottie import st_lottie
from transformers import pipeline
from typing import List


def encode_and_answer(question : str, docs : List[str] ) -> str:
    """
    Fucntion that encodes questions and documents, 
    performs sematic similarity search using bi-encider, and
    answers the provided question utilizing RoBERTa-model. 
    """

    encoder = SentenceTransformer('marticampgin/sciq_ft_bi_encoder')
    q_emb = encoder.encode(question, convert_to_tensor=True)
    doc_emb = encoder.encode(docs, convert_to_tensor=True, show_progress_bar=True)

    qa_model = pipeline('question-answering',
                        model='deepset/roberta-base-squad2',
                        tokenizer='deepset/roberta-base-squad2',
                        max_length=15)

    hit = util.semantic_search(q_emb, doc_emb, top_k=1)[0]  # Extract the top-1 hit
    
    return qa_model(question, docs[hit[0]["corpus_id"]])['answer']


def main() -> None:
    # Load animation
    with open('animation.json', 'r') as json_file:
        lottie_coding = json.load(json_file)

    # -------- Setting UI using streamlit --------
    st.set_page_config(page_title='Open Book QA', page_icon='U+1F9D9', layout='wide')

    with st.container():
        st.header('Sci. Q. Fine-tuned Bi-Encoder + QA-model 🦾')

    with st.container():
        st.write('---')
        left_column, right_column = st.columns(2)

        with left_column:
            st.subheader('Type a question, upload your document(s) (.txt) and get an answer from the relevant document!')
            question = st.text_input('Type in the question:')
            path = st.file_uploader('Choose files', accept_multiple_files=True)
            texts = [f.getvalue().decode('utf-8') for f in path]

            if st.button('Get answer'):
                st.write(f'Answer: {encode_and_answer(question, texts)}')
            
        with right_column:
            st_lottie(lottie_coding, height=300, key='robot')  # Load the animation

        
if __name__  == "__main__":
    main()
