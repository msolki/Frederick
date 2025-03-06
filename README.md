# Frederick - Course Description Guide Chatbot

Frederick is a chatbot designed to provide course descriptions in University of Naples Federico II. It is built using two web scrapers (one based on traditional scraping, one based on LLM scraping) and a RAG pipeline.

## Table of Contents
- [Usage](#usage)
- [API Keys](#api-keys)
- [Contributing](#contributing)
- [License](#license)
- [Resources](#resources)

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/msolki/Frederick.git
   cd Frederick
   ```
   
2. To start the chatbot, open the notebook in your browser and follow the instructions to interact with the chatbot.

## API Keys
The code retrieves API keys from environment variables or prompts the user to input them if they are not found. It supports both Google Colab and local environments.

You'll need to set `OPENAI_API_KEY` for the LLM scraper _(gpt-3.5-turbo)_, `groq_api_key` for the chatbot's LLM _(llama3-8b-8192)_  and, `huggingface_token` for the embedder _(all-MiniLM-L6-v2)_.

You are welcome to modify the models as you see fit to achieve better results.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes.

## License

This project is licensed under the Apache License 2.0.

## Resources 

- **Lewis et al. (2021)** – *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* ([arXiv](http://arxiv.org/abs/2005.11401)).  
- **ScrapeGraphAI** – Python library for scraping with LLMs ([GitHub](https://github.com/VinciGit00/Scrapegraph-ai)).  
- **all-MiniLM-L6-v2** – Sentence transformer for semantic search ([Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)).  
- **Reimers & Gurevych (2019)** – *Sentence-BERT: Siamese BERT-Networks* ([arXiv](http://arxiv.org/abs/1908.10084)).  
- **Comparison of Sentence Transformers** ([SBERT](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html#original-models)).  
- **Wang et al. (2020)** – *MiniLM: Self-Attention Distillation for Transformer Compression* ([arXiv](http://arxiv.org/abs/2002.10957)).  
- **Chroma** – AI-native open-source vector database ([LangChain](https://python.langchain.com/v0.2/docs/integrations/vectorstores/chroma/)).  
- **Meta Llama 3** – Most capable open LLM ([Groq](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), [Ollama](https://ollama.com/library/llama3)).  
- **LangChain Q&A with RAG** ([Docs](https://python.langchain.com/v0.1/docs/use_cases/question_answering/)).  
- **Galileo** – AI chatbot assistant for the University of Padova ([GitHub](https://github.com/sinadalvand/Galileo.git)).  
- **Wei et al. (2023)** – *Chain-of-Thought Prompting for LLM Reasoning* ([arXiv](http://arxiv.org/abs/2201.11903)).  
- **LangChain & Chain of Thought Prompting** ([Article](https://ai.plainenglish.io/langchain-in-chains-22-chain-of-thought-prompting-8b0dc4b01215)).  
