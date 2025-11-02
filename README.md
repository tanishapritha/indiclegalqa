# Indian Legal Question Answering System

An AI-powered legal question answering system trained on the IndicLegalQA dataset containing 10,000+ question-answer pairs from Indian Supreme Court judgments.

## Overview

This project fine-tunes DistilBERT using LoRA (Low-Rank Adaptation) to create an efficient retrieval-based QA system for Indian legal queries. The system searches through Supreme Court case precedents to provide relevant answers with confidence scores.

## Features

- Fine-tuned DistilBERT with 90% parameter reduction using LoRA
- Retrieval-based QA ranking 10,000+ legal question-answer pairs
- Sub-second inference time on CPU
- Confidence scoring for answer relevance
- Interactive web interface built with Streamlit
- Covers 1,256 Indian Supreme Court cases across criminal and civil law

## Model Performance

| Metric | Value |
|--------|-------|
| Evaluation Loss | 0.0001 |
| Throughput | 195 samples/second |
| Training Time | 45 minutes (T4 GPU) |
| Inference Time | <1 second (CPU) |
| Model Size | 250MB |
| Memory Usage | 800MB (inference) |

## Architecture

### Base Model Parameters

| Parameter | Value |
|-----------|-------|
| Model | DistilBERT-base-uncased |
| Total Parameters | 66M |
| Architecture | 6-layer Transformer |
| Hidden Size | 768 |
| Attention Heads | 12 |
| Vocabulary Size | 30,522 |
| Max Sequence Length | 512 |

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| LoRA Rank (r) | 8 |
| LoRA Alpha | 32 |
| Target Modules | q_lin, v_lin |
| LoRA Dropout | 0.1 |
| Trainable Parameters | 600K (0.9% of base) |
| Parameter Reduction | 90% |
| Task Type | Sequence Classification |

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 3e-4 |
| Batch Size (per device) | 16 |
| Gradient Accumulation Steps | 2 |
| Effective Batch Size | 32 |
| Epochs | 3 |
| Warmup Steps | 500 |
| Weight Decay | 0.01 |
| Max Sequence Length | 384 |
| Optimizer | AdamW |
| Mixed Precision | FP16 |

## System Architecture

```
User Question → DistilBERT (LoRA) → Semantic Scoring → Rank QA Pairs → Top-K Results
```

## Installation

### Prerequisites

- Python 3.8 or higher
- 8GB RAM minimum


Access the application at 'http://localhost:8501'

## Project Structure

```
legal-qa-app/
├── app.py                          # Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Documentation
└── model/                          # Model artifacts
    ├── adapter_config.json         # LoRA configuration
    ├── adapter_model.safetensors   # Trained weights
    ├── config.json                 # Model configuration
    ├── tokenizer files             # Tokenizer artifacts
    ├── dataset_full.json           # QA database (10K pairs)
    └── metadata.json               # Training metadata
```

## Technical Implementation

### Training Pipeline

1. **Data Preprocessing**
   - Load and validate JSON dataset
   - Format: 'Case: [name] | Date: [date] | Question: [question]'
   - Split: 80% train, 20% validation

2. **Tokenization**
   - Combine input and answer with [SEP] token
   - Max length: 384 tokens
   - Padding for consistent batching

3. **Model Configuration**
   - Base: DistilBERT-base-uncased
   - Apply LoRA to attention layers (q_lin, v_lin)
   - Task: Sequence classification with regression head

4. **Training**
   - Mixed-precision (FP16) training
   - Gradient accumulation
  
   
