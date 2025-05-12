# Transformer Text Classifier from Scratch

This project implements a transformer model for text classification using PyTorch. By building each component from scratch, you'll gain a deep understanding of how transformer architectures work.

## 🔍 Project Overview

This repository contains a complete implementation of a transformer encoder for text classification. Rather than using pre-built transformer modules, we build each component step-by-step to understand the internal workings of this powerful architecture.

## 🧠 What You'll Learn

- **Core Transformer Components**:
  - Positional encoding
  - Multi-head self-attention
  - Feed-forward networks
  - Layer normalization
  - Residual connections

- **Practical Implementation**:
  - How to process text data for transformer models
  - Training and evaluation procedures
  - Hyperparameter tuning for transformers

## 🛠️ Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy
- tqdm
- scikit-learn

Install dependencies:
```bash
pip install torch numpy tqdm scikit-learn
```

## 📋 Project Structure

```
transformer_text_classifier/
├── README.md                     # This file
├── transformer_classifier.py     # Main implementation file
├── requirements.txt              # Project dependencies
└── examples/                     # Example usage and outputs
```

## 🚀 Getting Started

1. Clone this repository:
```bash
git clone https://github.com/yourusername/transformer-text-classifier.git
cd transformer-text-classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the implementation:
```bash
python transformer_classifier.py
```

## 📊 The Dataset

The project uses the 20 Newsgroups dataset, a collection of approximately 20,000 newsgroup documents, partitioned across 20 different newsgroups. This provides a diverse set of topics for text classification.

## 🏗️ Model Architecture

The transformer implementation includes:

1. **Text Embedding Layer**: Converts tokens to vectors
2. **Positional Encoding**: Adds position information to token embeddings
3. **Transformer Encoder Layers**: 
   - Self-attention mechanism
   - Feed-forward networks
   - Layer normalization
   - Residual connections
4. **Global Pooling**: Aggregates sequence information
5. **Classification Head**: Maps to output classes

## ⚙️ Hyperparameters

The default hyperparameters are:

- Max Sequence Length: 128
- Vocabulary Size: 10,000
- Embedding Dimension: 128
- Number of Attention Heads: 4
- Feed-forward Dimension: 512
- Number of Transformer Layers: 2
- Dropout Rate: 0.1
- Batch Size: 64
- Learning Rate: 0.001
- Training Epochs: 5

Feel free to experiment with these parameters to see how they affect model performance.

## 🔧 Customization

You can modify various aspects of the transformer:

- Change the number of layers by adjusting `NUM_LAYERS`
- Increase or decrease model capacity with `EMB_SIZE` and `FF_DIM`
- Modify the attention mechanism by changing `NUM_HEADS`
- Experiment with different sequence lengths via `MAX_SEQ_LEN`

## 📈 Expected Results

With the default settings, you can expect:
- Training accuracy: ~85-90%
- Validation accuracy: ~65-70% 
- Test accuracy: ~65-70%

Results may vary slightly due to random initialization.

## 🔄 Extension Ideas

Here are some ways to extend this project:

1. Implement the decoder part of the transformer for sequence-to-sequence tasks
2. Add more sophisticated tokenization (e.g., subword tokenization)
3. Experiment with pre-training objectives like masked language modeling
4. Implement attention visualization to see what the model focuses on
5. Try different datasets or tasks like sentiment analysis

## 📖 Further Reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - The original transformer paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual guide to transformers
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) - Detailed implementation walkthrough

## 🙏 Acknowledgments

- The transformer architecture was introduced in "Attention Is All You Need" by Vaswani et al.
- The 20 Newsgroups dataset is a standard benchmark for text classification
