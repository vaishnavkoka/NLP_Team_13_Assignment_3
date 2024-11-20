## *<div align='center'>NLP_Assignment_2</div>*
### *<div align='center'> NLP_Team_13: Fine Tuning & Evaluation</div>*
<br><br>
### *Overview*
This project focuses fine-tuning and evaluation of Llama3.2-1B/Gemma model



<table>
  <tr>
    <th>Task 1: Training Tokenizers</th>
    <th>Task 2: Fine-Tuning Language Model</th>
  </tr>
  <tr>
    <td style="font-size: 80%;">1. Trained six different tokenizers on a 10k dataset.</td>
    <td style="font-size: 80%;">1. Adjusted the LLaMA model architecture to stay within a 100M parameter limit.</td>
  </tr>
  <tr>
    <td style="font-size: 80%;">2. Evaluated each tokenizer's performance using fertility scores.</td>
    <td style="font-size: 80%;">2. Used the top-performing tokenizer from Task 1 to tokenize the dataset.</td>
  </tr>
  <tr>
    <td style="font-size: 80%;">3. Selected the best-performing tokenizer based on fertility score results.</td>
    <td style="font-size: 80%;">3. Recorded perplexity scores at every 0.1 epoch.</td>
  </tr>
  <tr>
    <td style="font-size: 80%;"></td>
    <td style="font-size: 80%;">4. Created a matrix of perplexity scores for each epoch.</td>
  </tr>
  <tr>
    <td style="font-size: 80%;"></td>
    <td style="font-size: 80%;">5. Tested the fine-tuned model on 10 prompts to evaluate performance.</td>
  </tr>
</table>

<br><br><br>
### *Task Fine tuning and evaluation*
We trained six different tokenizers on samples from a previously scraped dataset to evaluate performance based on fertility scores:
- Each tokenizer was evaluated on fertility, and a matrix showing fertility scores and dataset size was generated.
- This matrix, available in [NLP-Assignment-2_Tokenizer_Training.pdf](https://github.com/vaishnavkoka/NLP_Assignment_2/blob/main/Results/NLP-%20Assignment-2_Tokenizer_Training.pdf), showed that **SentencePieceBPETokenizer** had the lowest fertility score and was selected as the optimal tokenizer for Task 2.
<br>

<table>
  <tr>
    <td width="66%">
      <img src="https://github.com/user-attachments/assets/1bcbeba8-9c55-4dd8-8151-eac9f08d13b2" width="100%">
    </td>
    <td width="33%">

| Tokenizer                        | Kaggle Link                               |
|-----------------------------------|-------------------------------------------|
| **BertWordPieceTokenizer**        | [🟩 Kaggle Link](https://www.kaggle.com/code/vaishnavkoka24310069/bertwordpiecetokenizer/) |
| **SentencePieceBPETokenizer**     | [🟩 Kaggle Link](https://www.kaggle.com/code/ramanand9/sentencepiecebpetokenizer/) |

   </td>
  </tr>
</table>
<br>

### *Dataset*

| Dataset                        | Kaggle Link                               |
|-----------------------------------|-------------------------------------------|
| **SST**        | [🟩 Kaggle Link](https://www.kaggle.com/datasets/atulanandjha/stanford-sentiment-treebank-v2-sst2) |
| **SQuAD2.0**     | [🟩 Kaggle Link](https://www.kaggle.com/datasets/thedevastator/squad2-0-a-challenge-for-question-answering-syst) |


<br><br><br>

### *Task 2.1: Model Training (LLaMA model used for training & evaluation)*

This project involves training a modified LLaMA model with fewer than 100M parameters for efficient training and evaluation. The model is tokenized using SentencePieceBPETokenizer.

- Tokenized dataset using SentencePieceBPETokenizer.
- Trained the model and logged *perplexity scores* at every 0.1 epoch.
- Generated a matrix of perplexity scores by epoch for analysis.
- Tested the model on 10 sample prompts.
- *Output screenshots* showcasing the model's performance are included in the repository.
<br>
<div align="center">
  <table style="display: inline-block; width: 100%; text-align: left; border: 2px solid black; border-collapse: collapse;">
    <tr>
      <th style="width: 33%; border: 2px solid black;">Weight</th>
      <th style="width: 33%; border: 2px solid black;">Link</th>
      <th style="width: 33%; border: 2px solid black;">Time</th>
    </tr>
    <tr>
      <td style="border: 2px solid black;">Transformer weight</td>
      <td style="border: 2px solid black;"><a href="https://www.kaggle.com/datasets/ramanand9/ps-v2-output/">🟩 Kaggle Link</a></td>
      <td rowspan="2" style="border: 2px solid black; text-align: center;">
        <img src="https://github.com/user-attachments/assets/4dab3cfd-0b45-4192-ae76-cd2efeac59db" alt="training_epoch" style="max-width: 80%;"/>
        <i>Model trained on 4.7M tokens for 10 epochs in 9 hours on 2 GPU T4</i>
      </td>
    </tr>
  </table>
</div>

<br><br>

<div align="center">
  <img src="https://github.com/user-attachments/assets/139e8689-b490-4076-9776-4986cf6d0071" alt="model_output_for_prompts" style="height: 250px; width: auto;"/>
  <br><i>Fig1: model training for every 0.1 epoch</i>
</div>

<br><br>

### *Task 2.2: Model Prediction*

<div align="center">
  <img src="https://github.com/user-attachments/assets/23f6160b-ecae-41ac-8953-12a117a5de9a" alt="model_output_for_prompts" style="height: 400px; width: auto;"/>
  <br><i>Fig3: Prompts output</i>
</div>


<br><br>

### *Contributions*
Each team member contributed significantly to both tasks, as detailed below:
- **Vaishnav Koka**: Created code for Tokenizers and trained tokenizers to calculate fertility scores, and Git Hub documentation.
- **Ramanand**: Created datasets, did Transformer model training and fine-tuning, and Git Hub documentation.
- **Isha Jain**: Created code for tokenizers and trained tokenizers to calculate fertility scores, and Git Hub documentation.
- **Yash Sahu**: Created tokenizers code, transformer fine-tuning, and Git Hub documentation.
<br>

### *Acknowledgments*
- Our team worked in a synchronous manner so as to not burden a single person, and get the work done more effectively and in a timely manner.
- Of course, it wouldn't have been possible without the teaching of Mayank sir.
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
10. https://huggingface.co/docs/datasets/v1.1.0/loading_metrics.html
