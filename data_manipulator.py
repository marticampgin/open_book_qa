import pandas as pd
import random
import numpy as np
from datasets import load_dataset
from typing import List, Tuple

class DataManipulator:
    def __init__(self, seed: int=7) -> None:
        self.sciq_dataset = load_dataset('sciq', split='train')  # Load sciq_dataset
        self.gpt_contexts = None
        self.gpt_contexts_and_qs = None
        random.seed(seed)


    def produce_training_data(self) -> Tuple[list]:
        df = self.sciq_dataset.to_pandas()
        df.drop(columns=['distractor3', 'distractor1', 'distractor2', 'correct_answer'], inplace=True)
        df.replace('', np.nan, inplace=True)
        df.dropna(how='any', inplace=True)

        # Only keep texts where both question and context are present (str is not empty)
        new_sci_q, new_sci_c = df['question'].tolist(), df['support'].tolist()
        
        # Retaining some of the contexts for later use in Retrieval AI https://relevanceai.com/ to generating data
        self.gpt_contexts = new_sci_c[:250]

        # Data for further fine-tuning the bi-encoder (first 250 contexts reserved)
        train_q_c = [{'context': context, 'question': question} 
                     for context, question in list(zip(new_sci_c[250:2750], new_sci_q[250:2750]))]
        
        # Shuffling data just to ensure that any potential order is broken
        random.shuffle(train_q_c)

        # Creating 'bad' (or rather neutral) and 'good' training samples
        good_training_data = []
        bad_training_data = []

        last_sample = None
        for sample in train_q_c:
            if last_sample and sample['context'] != last_sample['context']:
                # For question that don't align to contexts, the cosine-similarity should be 0.0 
                bad_training_data.append((sample['question'], last_sample['context'], 0.0))
            # For questions aligned with context, the cosine-similarity should be 1.0
            good_training_data.append((sample['question'], sample['context'], 1.0))
            last_sample = sample
        

        # Given, that we will have 250 questions generated for 250 examples,
        # we sample 2249 (2500 - 250 - 1) examples from good data (-1)
        # since there are 2499 bad data examples 
        sampled_training_data = random.sample(good_training_data, 2249) + random.sample(bad_training_data, 2499)
        random.shuffle(sampled_training_data)  # need to shuffle the data again

        return sampled_training_data
    
    def combine_gpt_and_training_data(self, gpt_data: List[tuple], train_data: List[tuple]) -> List[tuple]:
        combined_train_data = []
        combined_train_data.extend(train_data)
        for context, question in gpt_data:
            combined_train_data.append((question, context, 1.0))
        
        random.shuffle(combined_train_data)
        return combined_train_data
    

    def contexts_to_csv(self, path:str='sci_contexts.csv') -> None:
        df = pd.DataFrame(self.gpt_contexts, columns=['Contexts'])
        df.to_csv(path, index=False, encoding='utf-8')


    def csv_to_contexts(self, path:str='sci_contexts_and_questions.csv') -> None:
        df = pd.read_csv(path, encoding='charmap')
        df.dropna(inplace=True)  # Some texts got broken for some reason 
        self.gpt_contexts_and_qs = list(zip(df['Contexts'].tolist(), df['generated question'].tolist()))


    def get_gpt_contexts_and_qs(self) -> List[tuple]:
        return self.gpt_contexts_and_qs
    