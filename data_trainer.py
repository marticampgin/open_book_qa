from typing import List
from data_manipulator import DataManipulator
from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer, 
    losses,
    InputExample, 
    evaluation
)


class Datatrainer:
    """
    Simple wrapper around the bi-encoder fine-tuning.
    Simply made for modularity and easier reading. 
    """
    def __init__(self,
                 epochs: int=3,
                 train_data_ratio: float=0.8,
                 batch_size: int=64,
                 bi_encoder: str='msmarco-distilbert-cos-v5',
                 max_seq_len: int=128,
                 output_path: str='information_retrieval/results') -> None:
        
        # Init all the parameters
        self.bi_encoder = SentenceTransformer(bi_encoder)
        self.bi_encoder.max_seq_length = max_seq_len
        self.train_loss = losses.CosineSimilarityLoss(self.bi_encoder)
        self.train_data_ratio = train_data_ratio
        self.batch_size = batch_size
        self.epochs = epochs
        self.output_path = output_path
        self.data_manipulator = DataManipulator()


    def _prepare_data(self) -> List[tuple]:
        """
        Private method that prepares data by using data manipulator.
        The output data is combination of sci_qa dataset +
        private data (contexts) along with GPT generated questions. 
        """
        train_data = self.data_manipulator.produce_training_data()
        self.data_manipulator.csv_to_contexts()
        gpt_contexts_questions = self.data_manipulator.get_gpt_contexts_and_qs()
        train_data_combined = self.data_manipulator.combine_gpt_and_training_data(gpt_contexts_questions, train_data)
        return train_data_combined


    def train(self) -> None:
        """
        Simple method that prepares data for training loop before
        initiating it. fine-tuned model is saved to a provided path. 

        """
        train_data  = self._prepare_data()
        num_training_samples = int(self.train_data_ratio * len(train_data))  # Extracting % of data for training
        train_examples = [InputExample(texts=sample[:2], label=sample[2]) for sample in train_data[:num_training_samples]]
        train_dataloader = DataLoader(train_examples, batch_size=self.batch_size)  # Initiating pytorch-dataloader

        sent1, sent2, scores = zip(*train_data[num_training_samples:])
        evaluator =  evaluation.EmbeddingSimilarityEvaluator(sent1, sent2, scores)  # Initiating the evaluator

        print(f"Embeddings similarity before training: {self.bi_encoder.evaluate(evaluator)}")

        # Fine-tuning the model
        self.bi_encoder.fit(
            train_objectives=[(train_dataloader, self.train_loss)],
            output_path=self.output_path,
            epochs=self.epochs,
            evaluator=evaluator,
        )
        print("---------------------------------------")
        print(f"Embeddings similarity before training: {self.bi_encoder.evaluate(evaluator)}", end='\n\n')







