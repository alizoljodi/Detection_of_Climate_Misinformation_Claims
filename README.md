# Detection of Climate Misinformation Claims

## Overview
This project tackles the task of classifying climate-related claims into fine-grained sub-claim categories, following the taxonomy from Coan et al. (2021). Using the sub-claim configuration of the [murathankurfali/ClimateEval](https://huggingface.co/datasets/murathankurfali/ClimateEval/tree/main/exeter/sub_claim) dataset, we evaluate a spectrum of approaches in three phases: (1) zero-shot prompting, (2) few-shot prompting with limited in-context examples, and (3) advanced methods including PEFT QLoRA, Retrieval-Augmented Generation (RAG), and a TF-IDF + Fully Connected Neural Network (FNN) baseline. Due to resource constraints, few-shot prompting was restricted to a small set of representative examples, and fine-tuning was performed on a random subset of 1,000 training samples. Results show that naive prompting yields near-random performance, while PEFT QLoRA and RAG achieve substantial gains (≈72–74% accuracy). The conventional TF-IDF + FNN approach outperforms all LLM-based methods with 79% accuracy, highlighting the effectiveness of sparse lexical features in this domain. These findings suggest that for narrow, domain-specific classification tasks, traditional methods can surpass small language models unless they are domain-adapted or paired with strong retrieval pipelines.

The project is implemented in four phases:
- **Phase 0 - Exploratory Data Analysis**: Performing an overall dataset analysis to examine data types, distributions, and the most important features.
- **Phase 1 – Zero-Shot Prompting**: Establishing a baseline using direct prompts with no examples.
- **Phase 2 – Few-Shot Prompting**: Improving performance by adding in-context examples.
- **Phase 3 – Advanced Methods**: Exploring multiple approaches including PEFT with QLoRA, Retrieval-Augmented Generation (RAG), and TF-IDF + Fully Connected Neural Networks (FNN).

---

## Dataset
- **Source**: [murathankurfali/ClimateEval - sub_claim configuration](https://huggingface.co/datasets/murathankurfali/ClimateEval/tree/main/exeter/sub_claim)
- **Task**: Multi-class classification with a large number of categories (sub-claims).
- **Splits**: Train, validation, and test (test used strictly for final evaluation).

---
## Exploratory Data Analysis

1. **Data Integrity**  
   - Minimal redundancy: no overlaps between train and validation sets; only one duplicate in the test set.
  

| Split       | Metric    | text   | sub_claim_code | sub_claim |
|-------------|-----------|--------|----------------|-----------|
| **Train**   | Count     | 23,436 | 23,436         | 23,436    |
| **Test**    | Count     | 2,898  | 2,904          | 2,904     |
| **Validation** | Count | 2,605  | 2,605          | 2,605     |
| **Train**   | Unique    | 23,436 | 18             | 18        |
| **Test**    | Unique    | 2,897  | 18             | 18        |
| **Validation** | Unique| 2,605  | 18             | 18        |
| **Train**   | Top       | Truth n 16 The trace gases absorb the radiatio... | 0_0 | No claim |
| **Test**    | Top       | Reality may be a bit simpler, or much more com... | 0_0 | No claim |
| **Validation** | Top    | The linear trends in the graphs are as calcula... | 0_0 | No claim |
| **Train**   | Freq      | 1      | 16,302         | 16,302    |
| **Test**    | Freq      | 2      | 1,754          | 1,754     |
| **Validation** | Freq  | 1      | 1,812          | 1,812     |

   
2. **Label Distribution**  
   - Labels are highly imbalanced. The majority of examples belong to "No claim," which makes up ~⅔ of all data points.
   - This imbalance may bias models toward predicting the dominant class.

<p align="center">
  <img src="Figures/label_destribution.png" alt="Label distribution" width="600"/>
</p>


3. **Feature Characteristics**  
   - Text lengths vary significantly between samples, which may affect sequence models like LSTMs but less so large language models (LLMs).
   - Word count distribution shows a similar pattern to text length, confirming varied verbosity across examples.

4. **Implications for Modeling**  
   - Class imbalance requires careful consideration in evaluation.
   - Simple surface-level text features (e.g., TF-IDF) may already capture discriminative patterns given the specialized domain.

---
## Environment Setup

```bash
git clone https://github.com/alizoljodi/Detection_of_Climate_Misinformation_Claims.git
cd Detection_of_Climate_Misinformation_Claims
pip install -r requirements.txt
```

---

## Running the Code

Open the provided Colab notebook:

```bash
# For Google Colab
Open 'Detection_of_Climate_Misinformation_Claims.ipynb' in Colab
Follow the cell execution order for each phase.
```

You can run each phase independently:
- **Phase 1**: Zero-shot
- **Phase 2**: Few-shot
- **Phase 3**: PEFT QLoRA / RAG / TF-IDF + FNN

---

## Methods

### Phase 1 – Zero-Shot Prompting
- **Model**: Gemma 2B
- **Approach**: A simple classification prompt instructing the LLM to output exactly one label from the taxonomy.
- **Observation**: Accuracy was extremely low (≈0.1%), indicating that the small LLM had no prior alignment with this domain.

### Phase 2 – Few-Shot Prompting
- **Model**: Gemma 2B
- **Approach**: Added carefully selected examples from the training set to the prompt.
- **Observation**: Slight improvement over zero-shot but still far from acceptable performance.

### Phase 3 – Advanced Approaches
- **PEFT QLoRA**: Parameter-efficient fine-tuning of Gemma 2B with low-rank adapters.  
- **COT + Self-Consistency**: Chain-of-Thought reasoning with multiple answer paths (poor results in this domain).  
- **RAG**: Retrieval-Augmented Generation using the training set as a knowledge base, retrieving relevant examples before classification.  
- **TF-IDF + FNN**: Conventional machine learning with TF-IDF embeddings and a fully connected network classifier.

---

## Results

| Model / Method            | Accuracy  | Precision | Recall   | F1-Score |
|---------------------------|-----------|-----------|----------|----------|
| Zero-shot (Gemma 2B)       | 0.0010    | 0.6204    | 0.0010   | 0.0020   |
| Few-shot (Gemma 2B)        | 0.0055    | 0.2616    | 0.0055   | 0.0108   |
| PEFT QLoRA (Gemma 2B)      | 0.7198    | 0.6479    | 0.7198   | 0.6733   |
| COT + Self-Consistency     | 0.0000    | 0.0000    | 0.0000   | 0.0000   |
| RAG (Gemma 2B)             | 0.7380    | 0.7419    | 0.7380   | 0.7168   |
| TF-IDF + FNN               | **0.7919**| 0.7873    | 0.7919   | 0.7695   |

---

## Analysis and Reflections

The results show a clear gap between naive prompting and methods adapted to the task:

- **Zero-shot**: Poor performance due to the model's lack of task-specific knowledge and small size.
- **Few-shot**: Slight improvement from in-context learning, but still far from useful.
- **PEFT QLoRA**: Significant jump in performance, confirming that even partial fine-tuning can give the model useful domain knowledge.
- **RAG**: Outperformed fine-tuning slightly, showing the potential of retrieval-based augmentation in specialized tasks.
- **TF-IDF + FNN**: The top performer with 79% accuracy, demonstrating that conventional methods can outperform small LLMs in narrow, domain-specific classification tasks.

**Why TF-IDF + FNN succeeded**:  
- The task benefits from sparse, discriminative features.
- No need for deep semantic reasoning; surface-level patterns captured by TF-IDF suffice.
- Fully connected networks effectively learned non-linear decision boundaries from these features.

**Next Steps**:
- Use domain-adapted LLMs pre-trained on climate change discussions.
- Explore more advanced retrieval pipelines.

---

## Metrics

For this project, **accuracy** is used as the main comparison metric.  
The reason for this choice is its **simplicity** and ease of interpretation.  
Since this is a **multi-class classification** task, accuracy provides a straightforward  
yet effective way to compare models based on their overall detection performance.  
It also offers a quick summary of how well the model performs across all categories,  
making it useful for initial benchmarking before diving into more detailed metrics.

**Accuracy**  
Accuracy = Correct Predictions / Total Predictions

**Precision**  
Precision = True Positives / (True Positives + False Positives)

**Recall**  
Recall = True Positives / (True Positives + False Negatives)

**F1-Score**  
F1 = 2 × (Precision × Recall) / (Precision + Recall)

While precision, recall, and F1-score are also reported to capture the nuances of  
per-class performance and class imbalance effects, the primary focus remains on  
accuracy for clear model-to-model comparison.


---

## References
- Coan, T. G., Boussalis, C., & Cook, J. (2021). *Computer-assisted classification of contrarian claims about climate change*. *Scientific Reports, 11*(1), 1–12. [Link](https://www.nature.com/articles/s41598-021-01714-4)  
- [ClimateEval Dataset](https://huggingface.co/datasets/murathankurfali/ClimateEval/tree/main/exeter/sub_claim)  
- Team Google DeepMind & Google Research. (2024). *Gemma: Open models based on Gemini research and technology*. [Hugging Face model card](https://huggingface.co/google/gemma-2b)  
- Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Riedel, S. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. *Advances in Neural Information Processing Systems (NeurIPS)*. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)  
- Wei, J., Wang, X., Schuurmans, D., Bosma, M., Chi, E., Le, Q., & Zhou, D. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. *Advances in Neural Information Processing Systems (NeurIPS)*. [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)  
- Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., ... & Zhou, D. (2022). *Self-Consistency Improves Chain of Thought Reasoning in Language Models*. [arXiv:2203.11171](https://arxiv.org/abs/2203.11171)  
(https://www.nature.com/articles/s41598-021-01714-4)
- [ClimateEval Dataset](https://huggingface.co/datasets/murathankurfali/ClimateEval/tree/main/exeter/sub_claim)
