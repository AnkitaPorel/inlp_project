# Fine-Tuning Pretrained Language Models for Bengali Poetry Generation

**Authors:**  
Ankita Porel (2024201043)  
Swastik Pal (2024201011)  
Venkat Raghav S (2022101013)  

**Date:** May 06, 2025

---

## Abstract

This progress report details all the work we did for our project, *"Fine-Tuning Pretrained Language Models for Bengali Poetry Generation."* The project aims to enhance the capability of pretrained language models (e.g., Gemma3, GPT-2, Sarvam-AI) to generate culturally resonant and stylistically accurate Bengali poetry. We finetuned the Sarvam AI and Gemma 3-4b models on a poem dataset. Based on Manish Sir's suggestions, we added one more layer of fine-tuning before the poem dataset. This report outlines our objectives, completed tasks, challenges encountered, and next steps.

---

## 1. Introduction

Large-scale language models have shown remarkable success in text generation, yet their application to poetry or languages other than English remains limited—primarily due to scarce training data, complex poetic forms, and the language’s morphological richness. Our project seeks to address these challenges by fine-tuning a pretrained model on a curated Bengali poetry dataset. This report summarizes all the work we did, aligning with the timeline outlined in the project plan.

---

## 2. Objectives

The primary objectives of this project are:

- **Fine-tuning:** To fine-tune a pretrained language model (e.g., Gemma-3, GPT-2, Sarvam-AI) for generating coherent and stylistically accurate Bengali poetry.
- **Incorporating Poetic Structures:** To incorporate Bengali poetic structures, such as meter and rhyme, into the generated outputs.
- **Evaluation:** To evaluate the model’s performance using quantitative metrics (BLEU, ROUGE, CHRF) and qualitative human assessments for understanding rhyming sequences.

---

## 3. Datasets

1. **Bengali Poem Dataset:**  
   - Source: [GitHub](https://github.com/shuhanmirza/Bengali-Poem-Dataset)
   - Description: 6,070 poems of 137 poets from various genres, including classic and modern poetry. The dataset is well-structured, with metadata for each poem (e.g., title, poet, genre).

2. **Bengali Wiki Dataset:**  
   - Source: [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/bangla-wikipedia)

3. **Bengali Chat Dataset:**
   - Source:[Huggingface](https://huggingface.co/nari-labs/Dia-1.6B)

## 4. Work Log

### 4.1. Problem Understanding & Research (Week 1)

We have completed the following tasks:

- **Project Scope:** Defined the project scope—fine-tuning a pretrained model to generate Bengali poetry with cultural and linguistic fidelity.
- **Literature Review:** Studied key works such as:
  - *Pascual (2021)* on deep learning approaches for poetry generation.
  - *Acharya (2024)* on fine-tuning Gemma-2 for Bengali poetry.
  - *Stanford NLP (2024)* on prefix-control in multilingual poetry generation.
- **Poetic Structures:** Analyzed Bengali poetic structures, focusing on meter, rhyme, and grammatical norms.

---

### 4.2. Data Collection & Preprocessing (Week 1-2)

We have made significant progress in preparing the dataset:

- **Data Acquisition:**  
  Acquired the Bengali Poem Dataset from GitHub, containing 6,070 poems by 137 poets.

- **Preprocessing Steps:**  
  - **Tokenization:** Experimented with SentencePiece for breaking poems into words and subwords suitable for Bengali.  
  - **Normalization:** Converted text to lowercase and removed non-poetic elements (e.g., metadata, annotations).

- **Data Distribution:**  
  Ensured a balanced distribution of classic and modern poems for training and evaluation.

---

### 4.3. Model Selection

We evaluated pretrained models:

- **Sarvam AI:**  
  Considered because it is tailored for many Indian languages like Bengali, Hindi, Kannada, Malayalam, etc.
  
- **Gemma-3:**  
  Considered for potential adaptation to poetry generation in Bengali.

Initial experiments with these models were completed.

---

### 4.4. Finetuning on Poem Dataset

We finetuned the above pretrained models on a Bengali poem dataset, and we evaluated the performance of Sarvam AI on some metrics (discussed later in this report). We ran test cases too on all the pretrained models that we finetuned.

---

### 4.5. Unexpected Outputs in Mid Submission and Manish Sir's Suggestions

When we tested our finetuned models, sometimes their outputs had words that are not from the Bengali language. To get rid of this problem, Manish Sir suggested that we finetune our models in two phases instead of one. The first phase would be on a large Bengali text dataset, and the second phase would be on the poem dataset itself.

---

### 4.6. Finetuning Again

Based on the suggestions above, we finetuned the models again, in two phases. In the first phase, we tested on both the Bengali Wiki dataset and a Bengali Chat dataset.
For the second phase, we took the models obtained after finetuning in phase 1, and we trained those on the Bengali poem dataset from before.

---

## 5. Challenges Encountered

### 5.1. Data-related Challenges

- **Dataset Limitations:**  
  The Bengali Poem Dataset, while substantial, lacks diversity in certain poetic styles, which may affect model generalization.

### 5.2. Language-specific Challenges

- **Bengali Language Processing:**  
  Standard tokenizers struggle with Bengali's morphological complexity, necessitating experimentation with specialized tokenizers (e.g., Gemma3, Sarvam-AI).
- **Poetic Structure Modeling:**  
  Existing models have difficulty accurately capturing the intricacies of Bengali poetic structures.
- **Model Availability:**  
  There is a general scarcity of language models specifically optimized for Indian languages.

### 5.3. Computational Constraints

- **Limited GPU Resources:**  
  Fine-tuning large models required significant computational resources, constraining our ability to conduct extensive experimentation.
- **Training Efficiency:**  
  The first layer of fine-tuning (chat/wikidata) was particularly time-intensive, even after dataset splitting, due to large dataset sizes and limited GPU availability.

### 5.4. Model Output Issues

- **Mixed-language Generation:**  
  Our initial fine-tuning attempts resulted in unexpected outputs containing non-Bengali vocabulary, which necessitated the two-phase fine-tuning approach suggested by Manish Sir.

---

## 6. Initial Experimentation

### 6.1. Tokenizers

We experimented with a few custom tokenizers. The results are shown below (all tokenization was performed on the same sentence):

- **indic-bert Tokenizer**
- **sarvam-ai’s sarvam-1 Tokenizer**
- **gemma3- 1 billion parameters**
- **gemma3- 2 billion parameters**

**Discussion:**  
Tokenization with Sarvam-1 and Gemma3 (1B) gives a word-level tokenization, which suits general text-analysis tasks but struggles with unknown words and large vocabulary sizes. In contrast, Gemma3 (2B) and Indic-BERT appear to perform subword tokenization, which might be useful in capturing the morphological patterns in Bengali poetry (e.g., *Dhonyatak Shobdo*, *Onukar*).

*Check the tokenization results here.*

---

### 6.2. Sarvam AI (with and without fine-tuning)

We conducted initial experimentation with the Sarvam AI model and evaluated it using several metrics. Both the base version and a fine-tuned version of the model were examined. (The test was performed on a randomly selected 10% of the dataset, with the same test dataset applied across evaluations.)

**About the Test Metrics:**

- **BLEU (Bilingual Evaluation Understudy):**  
  Measures precision of n-gram overlap between generated and reference text. For a Bengali fine-tuned model, it assesses exact word matches (range: 0 to 1).

- **ChRF (Character n-gram F-score):**  
  Evaluates character-level n-gram similarity, balancing precision and recall (range: 0–100). This metric is ideal for capturing Bengali’s rich morphology.

- **ROUGE (Recall-Oriented Understudy for Gisting Evaluation):**  
  Computes recall-based n-gram overlap (ROUGE-1, ROUGE-2, ROUGE-L) (range: 0 to 1) to gauge content and sequence similarity—including poetic structure.

**Base Model Scores:**

- **ROUGE Scores:**  
  - ROUGE-1: 0.0016  
  - ROUGE-2: 0.0014  
  - ROUGE-L: 0.0016
- **BLEU Score:** 0.0171
- **ChRF Score:** 12.4741

---

### 6.3. Sarvam AI (with fine-tuning)

**Fine-Tuned Model Scores:**

- **ROUGE Scores:**  
  - ROUGE-1: 0.0032  
  - ROUGE-2: 0.0018  
  - ROUGE-L: 0.0026
- **BLEU Score:** 0.18510
- **ChRF Score:** 42.6665

*Link for the model: [Sarvam1 Fine-tuned Model](#)*

---

### 6.4. Interpreting the Results

- **ROUGE:**  
  A normalized ROUGE-1 score of around 0.5 or above is considered good; for ROUGE-2 and ROUGE-L, around 0.4 or above is desirable.

- **BLEU:**  
  A score around 0.5 or above is considered good on a normalized scale (0 to 1).

- **ChRF:**  
  A score of around 60 or above (on a scale from 0 to 100) is considered good.

The metrics improved considerably after fine-tuning; however, they still remain below acceptable cutoffs. This indicates that while fine-tuning has benefitted the performance, it is not yet sufficient for high-quality output.

---

### 6.5. Gemma-3 1B Fine-Tuning Results

**Disclaimer:** Due to time constraints on online GPU service platforms, the test metrics for this model are not provided as of now. They will be included in the final submission.

- **Link for Fine-Tuned Model:**  
  *Gemma3-1b-v1*  
  - **Output:** Github link to check the test runs

**Discussion:**  
The generated poems do not seem to follow the provided prompt title and category very well. At times, the output is incoherent, includes characters not native to Bengali, and lacks clear rhyming ends.

---

### 6.6. Gemma-3 4B Fine-Tuning Results

**Disclaimer:** Due to time constraints on online GPU service platforms, the test metrics for this model are not provided as of now. They will be included in the final submission.

- **Link for Fine-Tuned Model:**  
  *Gemma3-4b-v1*  
  - **Output:** Github link to check the test runs

**Discussion:**  
In this instance, the generated poems capture the provided prompt title and category considerably better. Although a few random characters appear in place of Bengali characters, the poems showcase more consistent and interesting patterns compared to previous experiments.

---

### 6.7. Visualize Weights and Biases in Training

- Training loss visualization for Gemma3-4b  
- System usage visualization for Gemma3-4b  
- Training loss visualization for Gemma3-1b

---

## 7. Fine-Tuning Methodology

Our fine-tuning approach leveraged the Unsloth library to efficiently adapt pre-trained models to Bengali poetry while addressing computational constraints.

### 7.1. Technical Implementation

We employed Parameter-Efficient Fine-Tuning (PEFT) using Low-Rank Adaptation (LoRA) to optimize the fine-tuning process:

```python
from unsloth import FastModel
import torch

model, tokenizer = FastModel.from_pretrained(
 model_name = "Ankita-Porel/gemma3-4B-pre-ft-v2", # depending on the model
 max_seq_length = 2048,
 load_in_4bit = True,
 load_in_8bit = False,
 full_finetuning = False,
)

model = FastModel.get_peft_model(
 model,
 r = 64,
 target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
       "gate_proj", "up_proj", "down_proj",
       "embed_tokens", "lm_head"], # Add for continual pretraining
 lora_alpha = 32,
 lora_dropout = 0, # Supports any, but = 0 is optimized
 bias = "none",    # Supports any, but = "none" is optimized
 use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
 random_state = 3407,
 use_rslora = True,
 loftq_config = None,
)
```

### 7.2. Prompt Engineering

We structured our dataset using a consistent prompt format in Bengali that included title, category, and poem fields:

```python
prompt = """আপনাকে নীচে উল্লিখিত স্টাইলে শিরোনাম সম্পর্কিত একটি কবিতা লেখার দায়িত্ব দেওয়া হয়েছে। কবিতাটি নির্দিষ্ট বিভাগের সাথে মানানসই হওয়া উচিত।

### Title:
{}
### Category:
{}

### Poem:
{}"""
```

This prompt template instructed the model to write a poem related to the specified title that matches the given category/style.

### 7.3. Training Configuration

We optimized our training pipeline with the following parameters to balance model performance with resource constraints:

```python
trainer = UnslothTrainer(
 model = model,
 tokenizer = tokenizer,
 train_dataset = dataset,
 dataset_text_field = "text",
 max_seq_length = max_seq_length,
 dataset_num_proc = 2,

 args = UnslothTrainingArguments(
  report_to = "wandb",
  per_device_train_batch_size = 2,
  gradient_accumulation_steps = 4,
  warmup_ratio = 0.1,
  num_train_epochs = 3,
  learning_rate = 5e-5,
  embedding_learning_rate = 1e-5,
  fp16 = not is_bfloat16_supported(),
  bf16 = is_bfloat16_supported(),
  logging_steps = 1,
  optim = "adamw_8bit",
  weight_decay = 0.01,
  lr_scheduler_type = "cosine",
  seed = 3407,
  output_dir = "outputs",
 ),
)
```

### 7.4. Key Technical Decisions

Several important technical choices influenced our results:

1. **Quantization:** We used 4-bit quantization to reduce memory requirements, enabling fine-tuning of larger models on limited GPU resources.

2. **LoRA Configuration:** We applied LoRA to multiple module types (attention, feed-forward, embeddings, and output layers) with a relatively high rank (r=64) to balance efficiency with model capacity.

3. **Learning Rate Differentiation:** We used a lower learning rate (1e-5) for embedding layers compared to other parameters (5e-5) to stabilize training.

4. **Two-Phase Approach:** For most configurations, we adopted a two-phase fine-tuning approach where we first fine-tuned on general Bengali text (Wiki or Chat) before specializing on poetry.

5. **Memory Optimization:** We implemented gradient checkpointing and 8-bit optimizer techniques to manage memory constraints when fine-tuning larger models.

The technical choices above influenced the evaluation results in the next section, particularly regarding the different responses of Sarvam AI and Gemma3-4b models to our fine-tuning approach.

---

## 8. Model Evaluation Results

| Base Model   | Fine-tune 1 | Fine-tune 2 | BLEU   | ROUGE-1 | ROUGE-2 | ROUGE-L | ChRF    |
|--------------|-------------|-------------|--------|---------|---------|---------|---------|
| Sarvam AI    | ×           | ×           | 0.0171 | 0.0016  | 0.0014  | 0.0016  | 12.4741 |
| Sarvam AI    | ×           | Poems       | 0.1851 | 0.0032  | 0.0018  | 0.0026  | 42.6665 |
| Sarvam AI    | Chat        | Poems       | 0.3795 | 0.0051  | 0.0044  | 0.0053  | 44.8507 |
| Sarvam AI    | Wiki        | Poems       | 0.3999 | 0.0072  | 0.0051  | 0.0072  | 46.5332 |
| Gemma3-4b    | ×           | ×           | 0.3779 | 0.0017  | 0.0017  | 0.0017  | 49.8204 |
| Gemma3-4b    | ×           | Poems       | 0.3525 | 0.0017  | 0.0011  | 0.0017  | 47.7409 |
| Gemma3-4b    | Chat        | Poems       | 0.3472 | 0.0083  | 0.0072  | 0.0081  | 45.2877 |
| Gemma3-4b    | Wiki        | Poems       | 0.3320 | 0.0026  | 0.0025  | 0.0027  | 44.7109 |

---

## 9. Analyzing the Results

Our comprehensive evaluation reveals several significant patterns and insights:

### 9.1. Sarvam AI Performance

- **Base to Fine-tuned Progression:** The Sarvam AI model shows dramatic improvement with fine-tuning. The base model performed poorly (BLEU: 0.0171, ChRF: 12.4741), but direct fine-tuning on poems increased scores substantially (BLEU: 0.1851, ChRF: 42.6665).
  
- **Two-phase Fine-tuning Benefit:** The two-phase approach suggested by Manish Sir proved highly effective. Wiki → Poems fine-tuning achieved the best overall performance for Sarvam AI (BLEU: 0.3999, ChRF: 46.5332), demonstrating that pre-training on general Bengali text before poetry-specific fine-tuning helps the model develop better linguistic representations.

- **Data Source Impact:** Wiki data provided slightly better results than Chat data as the first fine-tuning phase, possibly because Wikipedia text contains more formal and diverse language patterns that align better with poetic structures.

### 9.2. Gemma3-4b Performance

- **Strong Base Performance:** Surprisingly, the base Gemma3-4b model without any fine-tuning performed exceptionally well (BLEU: 0.3779, ChRF: 49.8204), suggesting it already has strong Bengali language capabilities.

- **Fine-tuning Regression:** Unlike Sarvam AI, fine-tuning actually decreased Gemma3-4b's performance across most metrics. This could indicate overfitting to the specific patterns in our poetry dataset, limiting the model's generalization capabilities.

- **Mixed ROUGE Results:** While BLEU and ChRF decreased with fine-tuning, ROUGE metrics showed improvement with the Chat → Poems pathway (ROUGE-1: 0.0083, ROUGE-2: 0.0072), suggesting better n-gram overlap despite potentially less coherent outputs overall.

### 9.3. Cross-Model Comparison

- **Base Model Capabilities:** Base Gemma3-4b significantly outperforms base Sarvam AI, indicating better pre-training on Bengali or related languages.

- **Fine-tuning Responsiveness:** Sarvam AI showed greater positive response to fine-tuning, with performance gains across all metrics, while Gemma3-4b's response was more nuanced or negative.

- **Optimal Configurations:** For Bengali poetry generation, our results suggest two viable pathways:
  1. Sarvam AI with Wiki → Poems fine-tuning (highest BLEU among fine-tuned models)
  2. Base Gemma3-4b without fine-tuning (highest ChRF overall)

### 9.4. Metric Analysis

- **BLEU vs. ChRF:** While BLEU focuses on precision of word matches, ChRF's character-level assessment may better capture Bengali's morphological richness. The divergence between these metrics in some cases suggests different aspects of generation quality.

- **Low ROUGE Scores:** Despite improvements, ROUGE scores remain relatively low across all models, indicating that generating exact n-gram matches with reference texts remains challenging in creative poetry generation.

### 9.5. Implications for Bengali Poetry Generation

- **Two-phase Approach Validation:** Our results validate the two-phase fine-tuning approach for Sarvam AI, confirming that general language understanding followed by domain-specific tuning produces better poetry.

- **Pre-training Importance:** Gemma3-4b's strong base performance highlights the importance of comprehensive pre-training on diverse multilingual data.

- **Balancing Creativity and Accuracy:** The tension between metric improvements and potential overfitting suggests the need to balance statistical similarity with creative expression in poetry generation systems.

## 10. Conclusion and Future Work
