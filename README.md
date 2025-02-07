# LLM Tools

A collection of scripts and utilities that leverage Large Language Models (LLMs) and AI-driven techniques. These tools aim to showcase various ways to integrate, optimize, and experiment with advanced machine learning concepts.

## Contents

- **Hybrid Recommendation System**  
  Combines collaborative filtering with content-based methods (TF-IDF). Includes hyperparameter tuning using [Optuna](https://optuna.org/). Demonstrates end-to-end flow: data generation, model training, validation, and testing.

- **TinyGPT**  
  A minimalist from-scratch transformer language model that dynamically scrapes raw online text, builds its own Byte Pair Encoding (BPE) tokenizer, and trains a lean model using CUDA AMP mixed precision.
  
- **Additional Scripts**  
  Future scripts will be added here to explore different LLM or AI-driven functionalities.

## Getting Started

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/0xMarcio/llm-tools
   ```
2. **Install Dependencies**  
   ```bash
   cd llm-tools
   pip install -r requirements.txt
   ```
3. **Run Scripts**  
   Use `python3 <script_name>.py` to execute any script within this repository.

## Contributing

1. Fork this repository.
2. Create a new branch for your feature or fix.
3. Commit your changes and open a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
