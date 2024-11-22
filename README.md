## *<div align='center'>NLP_Assignment_3</div>*
### *<div align='center'> NLP_Team_13: Fine Tuning & Evaluation</div>*


### *Overview*
This project aims to fine-tune the pre-trained Llama3.2-1B/Gemma model for two specific Natural Language Processing (NLP) tasks: Sentiment Classification (SST-2) and Question-Answering (SQuAD). The goal is to evaluate the model’s performance before and after fine-tuning and explore how fine-tuning enhances its ability to handle task-specific requirements.

### *Tasks:*
**Sentiment Classification (SST-2)**: This binary classification task involves predicting the sentiment of a sentence as either positive or negative.
**Question-Answering (SQuAD)**: This task involves answering a question based on a provided context from the SQuAD dataset.
By fine-tuning the model on these tasks, we aim to improve its ability to perform classification and question-answering accurately.

[```Flow of this documentation is as per the tasks mentioned in this paper```](https://docs.google.com/document/d/1uU6isq4UAufFzfoKtZmzoHl01X2IDBFe1qrcf4SNDxY/edit?tab=t.0])
<br><br>
### **Task-1):-** *Selection of model*
**For this study we chose both models;**
- Gemma for SST-2 dataset classification task
- Llama for Squad-2 Q&A task.
<br><br>
### **Task-2):-** *Number of Parameters in Gemma & Llama-3.2-1b*

![image](https://github.com/user-attachments/assets/d6825bd2-dec5-44c7-8a64-158f78a622d0)



### Fine-Tuning Process
The model was fine-tuned on two datasets:

## 1. SST-2 (Sentiment Classification):

The SST-2 dataset from the GLUE benchmark was used to fine-tune the model for binary sentiment classification.
The dataset was split into 80% training and 20% testing.
The model was trained for 3 epochs, using a learning rate of 2e-5 and a batch size of 8.

## 2. SQuAD (Question-Answering):

The SQuAD dataset was used to fine-tune the model on the task of answering questions from a provided context.
The dataset was similarly split into 80% training and 20% testing.
The model was trained with the same hyperparameters as for SST-2 to ensure consistency across tasks.
Both tasks were evaluated on their respective test splits, and the fine-tuned model was compared to the pre-trained (zero-shot) model to observe the improvements in performance


### Metrics
The following evaluation metrics were used to assess the model’s performance:

## For SST-2 (Sentiment Classification):
**Accuracy**: Measures the overall correctness of the predictions.

**Precision**: The proportion of true positive predictions among all positive predictions.

**Recall**: The proportion of true positive predictions among all actual positive instances.

**F1 Score**: The harmonic mean of precision and recall, providing a balance between the two.


## For SQuAD (Question-Answering):
**Exact Match (EM)**: The percentage of answers that exactly match the ground truth answer.

**F1 Score**: The harmonic mean of precision and recall for each question-answer pair.

**METEOR**: Measures the alignment between generated and reference answers using synonymy and stemming.

**BLEU**: A metric for evaluating the precision of n-grams in the generated answers.

**ROUGE**: Measures the overlap of n-grams between the predicted and reference answers.

These metrics provide a comprehensive evaluation of the model’s performance on each task and help determine the effectiveness of fine-tuning.

## Push to Hugging Face 🤗
After fine-tuning, the model was uploaded to the **Hugging Face** Model Hub for easy access and sharing with the community. The fine-tuned model is now available for others to use for sentiment classification and question-answering tasks.

## Points to ponder
### Lower or Higher Scores in the Metrics

-Fine-tuning generally leads to higher scores on task-specific metrics (such as accuracy for classification and exact-match for question-answering).

-The improvement depends on the quality of the fine-tuning dataset, the size of the model, and the training process. If the fine-tuning dataset is small or poorly labeled, the performance gains may be minimal.

### Understanding from the Number of Parameters

-Number of parameters typically remains constant during fine-tuning because we are not altering the architecture of the model. Fine-tuning involves updating the weights of the existing parameters based on task-specific data, not adding or removing any parameters. The total count of 1 billion parameters in the Llama3.2-1B/Gemma model remains unchanged.

### Performance Difference for Zero-Shot and Fine-Tuned Models

-The fine-tuned model performs better on task-specific tasks like SST-2 and SQuAD compared to the zero-shot model. This is because fine-tuning helps the model specialize in the target task by adjusting its weights based on labeled data. The zero-shot model can make reasonable predictions but lacks the specialized knowledge gained through fine-tuning.



| Model                        | Kaggle Link                               |
|-----------------------------------|-------------------------------------------|
| **Llama using SST Dataset**        | [🟩 Kaggle Link](https://www.kaggle.com/code/vaishnavkoka24310069/fine-tuned-llama-sst/) |
| **SentencePieceBPETokenizer**     | [🟩 Kaggle Link]() |

![llama pre training parameters](https://github.com/user-attachments/assets/063aa203-6d1d-45c8-917d-e38b8d98ffdd)
![llama pre training metrics](https://github.com/user-attachments/assets/9fbd1122-47da-4fa8-b870-f0b77c69e096)
![llama post training parameters](https://github.com/user-attachments/assets/942f1599-b396-4ab8-b294-bc40608c64af)
![llama post training metrics](https://github.com/user-attachments/assets/d1a7024d-4efc-4a63-8e11-f81b5f09b2bd)



### *Dataset*

| Dataset                        | HuggingFace Link                               |
|-----------------------------------|-------------------------------------------|
| **SST**        | [🤗 HuggingFace Link](https://huggingface.co/datasets/stanfordnlp/sst2) |
| **SQuAD2.0**     | [🤗 HuggingFace Link](https://huggingface.co/datasets/rajpurkar/squad_v2) |


<br><br><br>

This README serves as a detailed explanation of the project, from the model selection to fine-tuning, evaluation, and performance analysis. It should help other users understand the purpose and methodology behind the fine-tuning process, as well as how to use the fine-tuned model.







### *Contributions*
Each team member made significant contributions to the project tasks, as detailed below:

**Vaishnav Koka:**

-Developed the code for tokenizers and trained tokenizers to calculate fertility scores.

-Implemented fine-tuning for the SST-2 classification task and contributed to the evaluation metrics for classification.

-Managed and updated the GitHub documentation to reflect the changes and progress made throughout the project.

**Ramanand:**
-Worked on the Transformer model training and fine-tuning for the SQuAD question-answering task.

-Contributed to evaluating the model's performance using the appropriate metrics (e.g., F1, Exact-Match, BLEU, ROUGE) for the question-answering task.

-Assisted with organizing and enhancing the GitHub documentation.

**Isha Jain:**
-Developed and trained tokenizers to calculate fertility scores.

-Fine-tuned the Llama3.2-1B/Gemma model for both SST-2 and SQuAD tasks.

-Contributed to the implementation of metrics for both tasks and ensured proper documentation was created for the project.


**Yash Sahu:**
-Focused on fine-tuning the Transformer model for both the SST-2 and SQuAD tasks, ensuring the proper train-test split and evaluation.

-Contributed to calculating and analyzing the performance metrics for both tasks (classification and question-answering).

-Supported the GitHub documentation by maintaining the structure and ensuring the process was clearly outlined.


### *Acknowledgments*
-Our team collaborated synchronously to distribute the workload evenly and avoid overburdening any single member.
-By working together in real-time, we ensured that tasks were completed more effectively and within the set timelines.
-The successful completion of this project was made possible with the invaluable guidance and teaching of Mayank Sir.
<br>



*References:*
1. https://huggingface.co/meta-llama/Llama-3.2-1B
2. https://huggingface.co/google/gemma-2-2b-it
3. https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
4. https://www.kaggle.com/datasets/atulanandjha/stanford-sentiment-treebank-v2-sst2/code
5. https://huggingface.co/datasets/rajpurkar/squad_v2
6. https://discuss.huggingface.co/t/how-to-split-hugging-face-dataset-to-train-and-test/20885/3
7. https://huggingface.co/docs/evaluate/en/choosing_a_metric
8. https://huggingface.co/tasks/question-answering
9. https://anthonywchen.github.io/Papers/evaluatingqa/mrqa_slides.pdf
10. https://medium.com/@sabaybiometzger/fine-tuning-gemma-2b-for-binary-classification-4-bit-quantization-60437e877723
11. https://huggingface.co/docs/datasets/v1.1.0/loading_metrics.html
