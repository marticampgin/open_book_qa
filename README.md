# Open Book Scientific Question Answering 🤖
In this project, the goal was to create an Open Book QA-system that will specifically be tailored to answer scientific questions.

It is lightweight (compared to generative LLMS) and extracts relevant answers from provided documents. Everything is wrapped in a simple
Streamlit UI for more user-friendly usage. 

## Usage:
```
pip install -r requirements.txt
streamlit run app.py
```
## How is it working?

### Generating data
The dataset used for the project contained scientific questions, contexts, and respective answers (https://huggingface.co/datasets/sciq).

Moreover, I simulated a 'lack of data' by taking out a portion of contexts without their questions. This was done in order to try and
generate more data using gpt-3.5-turbo model from OpenAI, by sending inputs in bulk utilizing Relevance AI (https://relevanceai.com/).

![alt text](https://github.com/marticampgin/open_book_qa/blob/main/rel_img/prompt.png)

Here, we add a prompt to gpt-model and also create a variable in a form of a context, which are stored in a .csv file. 

### Fine-tuning models & pipeline
After generating some extra data, a distill-bert-based bi-encoder was further fine-tuned on scientific questions and contexts for semantic search. 
Given that the model is already fine-tuned on msmarco-dataset, further fine-tuning didn't improve the metrics too much, but still the model
became slightly more confident in associating a specific scientific question with its correct context.

The next step was to assemble everything in one pipeline, together with a QA-model based on roberta model, fine-tuned on the squadv2 dataset.

