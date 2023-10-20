import streamlit as st
import json

from sentence_transformers import SentenceTransformer, util
from streamlit_lottie import st_lottie
from transformers import pipeline
from typing import List


def extract_document(docs: List[str], question: str) -> str:
    model_checkpoint = 'marticampgin/sciq_ft_bi_encoder'
    model = SentenceTransformer(model_checkpoint)  # Loading the model
    document_embeddings = model.encode(docs, convert_to_tensor=True, show_progress_bar=True)
    question_embedding = model.encode(question, convert_to_tensor=True)

    hit = util.semantic_search(question_embedding, document_embeddings, top_k=1)[0][0]
    doc = docs[hit['corpus_id']]
    
    with open('doc.txt', 'w') as document:
        document.write(doc)


def answer_question(question: str, context: str) -> None:
    # Init QA-model
    qa_model = pipeline('question-answering',
                         model='deepset/roberta-base-squad2',
                         tokenizer='deepset/roberta-base-squad2',
                         max_length=15)
    
    answer = qa_model(question, context)
    
    with open('ans.txt', 'w') as ans_file:
        ans_file.write(answer['answer'])

    
def main() -> None:
    if 'clicked' not in st.session_state:
        st.session_state.clicked = False

    with open('animation.json', 'r') as json_file:
        lottie_coding = json.load(json_file)

    # Setting UI using streamlit
    st.set_page_config(page_title="Open Book QA", page_icon='🧙🏻‍♂️', layout='wide')

    with st.container():
        st.header("Sci_Q Fine-tuned Bi-Encoder + QA-model 🦾")

    with st.container():
        st.write('---')
        left_column, right_column = st.columns(2)

        with right_column:
            st_lottie(lottie_coding, height=300, key='robot')

        with left_column:
            scientific_question = st.text_input('Enter a scietific question')
            doc1 = st.text_input('Enter doc 1. ')
            doc2 = st.text_input('Enter doc 2. ')
            doc3 = st.text_input('Enter doc 3. ')
            doc_button = st.button('Retreive doc', 
                                   on_click=extract_document, 
                                   args=([doc1, doc2, doc3], scientific_question))
            
            if doc_button:
                with open('doc.txt', 'r') as document:
                    context = document.read()
                st.write(f'Retrieved document: {context}') 


                q_button = st.button('Answer the question', 
                                        on_click=answer_question,
                                        args=(scientific_question, context))
                    
                if q_button:
                    print('suka')
                    with open('ans.txt', 'r') as ans_file:
                        answer = ans_file.read()
                    st.write(f'Answer: {answer}')
                    print('sukla')


if __name__  == "__main__":
    main()
