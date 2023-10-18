from typing import List
from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer, 
    losses,
    InputExample, 
    evaluation
)

class Datatrainer:
    def __init__(self,
                 data: List[tuple],
                 epochs: int=3,
                 train_data_ratio: float=0.8,
                 batch_size: int=64,
                 bi_encoder: str='msmarco-distilbert-cos-v5',
                 max_seq_len: int=128) -> None:
        
        self.bi_encoder = SentenceTransformer(bi_encoder)
        self.bi_encoder.max_seq_length = max_seq_len
        self.train_loss = losses.CosineSimilarityLoss(self.bi_encoder)
        self.data = data
        self.train_data_ratio = train_data_ratio
        self.batch_size = batch_size
        self.epochs = epochs


    def train(self):
        num_training_samples = int(self.train_data_ratio * len(self.data))  # Produce 80/20 train/test split
        train_examples = [InputExample(texts=sample[:2], label=sample[2]) for sample in self.data[:num_training_samples]]
        train_dataloader = DataLoader(train_examples, batch_size=self.batch_size)

        sent1, sent2, scores = zip(*self.data[num_training_samples:])
        evaluator =  evaluation.EmbeddingSimilarityEvaluator(sent1, sent2, scores)

        print(f"Embeddings similarity before training: {self.bi_encoder.evaluate(evaluator)}")

        self.bi_encoder.fit(
            train_objectives=[(train_dataloader, self.train_loss)],
            output_path='information_retrieval/results',
            epochs=self.epochs,
            evaluator=evaluator,
        )
        print("---------------------------------------")
        print(f"Embeddings similarity before training: {self.bi_encoder.evaluate(evaluator)}")







