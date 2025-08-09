# Detection of Climate Misinformation Claims

## Overview
This project addresses the task of classifying climate-related claims into their correct sub-claim categories, based on the taxonomy from Coan et al. (2021). The dataset is sourced from [murathankurfali/ClimateEval](https://huggingface.co/datasets/murathankurfali/ClimateEval/tree/main/exeter/sub_claim), focusing on the *sub-claim* configuration.

The project is implemented in three phases:
1. **Phase 1 – Zero-Shot Prompting**: Establishing a baseline using direct prompts with no examples.
2. **Phase 2 – Few-Shot Prompting**: Improving performance by adding in-context examples.
3. **Phase 3 – Advanced Methods**: Exploring multiple approaches including PEFT with QLoRA, Retrieval-Augmented Generation (RAG), and TF-IDF + Fully Connected Neural Networks (FNN).

---

## Dataset
- **Source**: [murathankurfali/ClimateEval - sub_claim configuration](https://huggingface.co/datasets/murathankurfali/ClimateEval/tree/main/exeter/sub_claim)
- **Task**: Multi-class classification with a large number of categories (sub-claims).
- **Splits**: Train, validation, and test (test used strictly for final evaluation).

---

## Environment Setup

```bash
git clone <your-repo-url>
cd <your-repo-name>
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
- Experiment with hybrid methods (e.g., TF-IDF features + LLM embeddings).
- Use domain-adapted LLMs pre-trained on climate change discussions.
- Explore more advanced retrieval pipelines.

---

## Metrics

**Accuracy**  
Accuracy = Correct Predictions / Total Predictions

**Precision**  
Precision = True Positives / (True Positives + False Positives)

**Recall**  
Recall = True Positives / (True Positives + False Negatives)

**F1-Score**  
F1 = 2 × (Precision × Recall) / (Precision + Recall)

---

## References
- Coan, T. G., Boussalis, C., & Cook, J. (2021). *Computer-assisted classification of contrarian claims about climate change*. *Scientific Reports, 11*(1), 1–12. [Link](https://www.nature.com/articles/s41598-021-01714-4)
- [ClimateEval Dataset](https://huggingface.co/datasets/murathankurfali/ClimateEval/tree/main/exeter/sub_claim)
