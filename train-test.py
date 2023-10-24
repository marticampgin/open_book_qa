import pandas as pd

from transformers import pipeline
from data_trainer import Datatrainer
from data_manipulator import DataManipulator
from sentence_transformers import SentenceTransformer, util


# SCRIPT FOR TESTING PURPOSES


def start_training() -> None:
    """
    Start training.
    """
    data_trainer = Datatrainer()
    data_trainer.train()
    

def test() -> None:
    """
    Tests the fine-tuned bi-encoder & QA-model.
    """

    # Init QA-model
    qa_model = pipeline('question-answering',
                        model='deepset/roberta-base-squad2',
                        tokenizer='deepset/roberta-base-squad2',
                        max_length=15)
    
    finetuned_bi_encoder = SentenceTransformer('information_retrieval/results')  # Load fine-tuned bi_encoder

    # Some data for testing
    dm = DataManipulator()
    dataset = dm.sciq_dataset.to_pandas()
    docs = dataset['support'][:-50].tolist()  # Last 50 contexts haven't been used in training
    questions = dataset['question'][:-1].tolist()  # Last question

    # Encode all docs. and one of the questions
    document_embeddings = finetuned_bi_encoder.encode(docs, convert_to_tensor=True, show_progress_bar=True)
    question_1_embedding = finetuned_bi_encoder.encode(questions[0], convert_to_tensor=True)
    
    hits = util.semantic_search(question_1_embedding, document_embeddings, top_k=3)[0]  # Extracting top 3 hits
    
    # Print question, top 3 hits
    print(f'Question: {questions[0]}\n')
    for i, hit in enumerate(hits):
        print(f'Document {i + 1} cos_sim {hit["score"]:.2f}:\n\n{docs[hit["corpus_id"]]}')
        print('\n')

    # Run the question and the top 1 hit through the QA-model
    print(f'Answer from Top Document: {qa_model(questions[0], str(docs[hits[0]["corpus_id"]]))}')
        

def main() -> None:
    # Uncomment for fine-tuning
    # start_training()
    test()


if __name__ == '__main__':
    main()