# LLAMA2RASA
Rasa platform Integration with open source LLM

Tested with RASA 2.8

Python environments needed:
- for RASA and Action Server
- for FastApi, LLama.cpp and LangChain

Client: https://github.com/teeso/Chatbot-Widget

Commands:
- Start RASA: rasa run --enable-api --cors "*"
- Start Action Server: rasa run actions
- Start model API: uvicorn driver:app