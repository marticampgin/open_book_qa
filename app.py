from transformers import pipeline
from data_manipulator import DataManipulator
from data_trainer import Datatrainer
from sentence_transformers import SentenceTransformer
def main():

    qa_model = pipeline('question-answering',
                        model='deepset/roberta-base-squad2',
                        tokenizer='deepset/roberta-base-squad2',
                        max_length=15)
    
    data_manipulator = DataManipulator()
    training_data = data_manipulator.produce_training_data()

    data_manipulator.csv_to_contexts()

    gpt_contexts_questions = data_manipulator.get_gpt_contexts_and_qs()
    train_data_combined = data_manipulator.combine_gpt_and_training_data(gpt_contexts_questions, training_data)
  
    data_trainer = Datatrainer(train_data_combined)
    data_trainer.train()
    
    finetuned_bi_encoder = SentenceTransformer('information_retrieval/results')

    


if __name__ == '__main__':
    main()