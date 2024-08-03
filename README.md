# RASA Open Source with Llama LLM integration

## Idea:
With over 25 million downloads, Rasa Open Source is the most popular open source framework for building chat and voice-based AI assistants.

The Rasa platform works great with business scenarios and integrates with external services. But its intelligence is limited by NLU capabilities. When the platform cannot determine the user's intent, an NLU fallback occurs and the user receives a standard phrase asking to rephrase the question. In this project, the NLU fallback causes an appeal to the LLM. In addition, RASA can access the knowledge base with RAG mechanics both with questions that are recognized by NLU as questions to the knowledge base, and with subsequent questions from which the context does not clearly follow. In this case, memory Slots are used, individual for each user and implemented as a basic part of the platform functionality.

The results of the project can be used to expand the functionality of AI assistants in business operation.

## Features of implementation:

The project uses three types of LLM calls:
1. In case of NLU fallback if the knowledge base has not yet been used (**free question**). Using LLM in free mode is unlikely to be used in a commercial platform.
2. Accessing the knowledge base with an intent known to NLU (**standalone question**). 
3. An access to the knowledge base with a question from which the context is unclear and it is not recognized by NLU, while the user has already received an answer from the knowledge base, and the context is saved in the memory slot (**follow up question**). In this case, the LLM must reformulate the follow up question into standalone question. Few-shot prompting technique is used.

## Basic libraries and modules:
- Rasa Open Source 2.8.0
- Rasa SDK 2.8.6
- Spacy 3.1.3 + ru-core-news-md 3.1.0 (for Rasa NLU)
- langchain 0.1.13
- langchain-community 0.0.29
- langchain-core 0.1.45
- REST framework: FastAPI 0.110.1
- Backend: llama_cpp_python 0.2.57 https://github.com/ggerganov/llama.cpp

The code that makes the model work is in the file **model.py**

## Technical features:

To run the project, you need two Python environments. One for the Rasa Core and Rasa Action Server (Python 3.8), the other for the FastAPI and llama.cpp backend (Python 3.10).

The markdown text of the knowledge base for experiments was taken from the repository https://github.com/sicutglacies/llm_rag.git

- Vector store: chromadb 0.4.24
- LLM: TheBloke/saiga_mistral_7b-GGUF quantized for 8 bits.
- Embedding: cointegrated/LaBSE-en-ru

## Installation and deployment:

Rasa installation: https://rasa.com/docs/rasa/2.x/installation

After installation, you need to train Rasa with data from the repository:
`rasa train`

Start Rasa:
` rasa run --enable-api --cors "*"`

Start Rasa Action Server:
`rasa run actions`

Start FastAPI with LLM:
`uvicorn driver:app`

Recommended client for tests and interactions: https://github.com/teeso/Chatbot-Widget

To run the chatbot with LLM without noticeable delay, 12+  GB GPU is required, however, it will work without an accelerator.