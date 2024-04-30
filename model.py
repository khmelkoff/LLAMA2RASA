from langchain_community.llms import LlamaCpp
from langchain.schema import StrOutputParser
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from typing import List, Tuple, Final
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from operator import itemgetter


# Settings
MODEL_NAME = 'saiga_mistral_7b_q8_0'
KB_NAME = 'Банковские продукты'
KB_FILE_PATH = 'kb/bank_name_docs2.md'
HEADERS_TO_SPLIT: Final = [
    ("#", "Заголовок темы"),
    ("##", "Подзаголовок темы"),
    ("###", "Заголовок абзаца")
]
PROMPT_TEMPLATE = """
<s>"role": "system"\n"content": Ты — ассистент, отвечаешь на вопросы. Всегда отвечай с сарказмом и юмором. 
Если не знаешь ответ, попроси переформулировать вопрос."</s>
<s>"role": "user"\n"content": {input}</s>
<s>bot\n
"""
PROMPT_TEMPLATE_KB = """
<s>"role": "system"\n"content": "Вы помощник по продуктам банка bank_name и отвечаете на вопросы клиентов.
Используйте фрагменты полученного контекста, чтобы ответить на вопрос.
Если вы не знаете ответа, то скажите, что не знаете, не придумывайте ответ.
Используйте максимум три предложения и будьте краткими.\n Контекст: {context} "</s><s>"role": "user"\n"content": Вопрос: {question}</s>
<s>bot\n Ответ:
"""
BM25_K = 1
MMR_K = 2
MMR_FETCH_K = 3


# Text loader and title splitter
def load_and_split_markdown(file_path: str, splitter: List[Tuple[str, str]]):
    loader = TextLoader(file_path, encoding='utf8')
    docs = loader.load()

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=splitter)
    md_header_splits = markdown_splitter.split_text(docs[0].page_content)
    return md_header_splits


# Knowledge Base Retriever
def get_retriever(splits, bm25_k, mmr_k, mmr_fetch_k):

    # Embeddings for vector search
    embedding = HuggingFaceEmbeddings(
        model_name="cointegrated/LaBSE-en-ru", model_kwargs={"device": "cuda"}
    )

    # DB for our vectors
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)

    # Key-word retriever
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = bm25_k

    # Vector-based retriever
    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={'k': mmr_k, 'fetch_k': mmr_fetch_k}
    )

    # Retriever combination
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, mmr_retriever],
        weights=[0.4, 0.6]
    )

    return ensemble_retriever


# Retrieved documents formatter
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


docs = load_and_split_markdown(KB_FILE_PATH, HEADERS_TO_SPLIT)
print('Knowledge base loaded, kb chancks:', len(docs))

ensemble_retriever = get_retriever(
    docs,
    BM25_K, MMR_K, MMR_FETCH_K
)
print('Retriever created')


class Model:

    def __init__(self) -> None:
        self.converse = None
        self.model_name = MODEL_NAME
        self.kb_name = KB_NAME

        model_path = "llm/" + self.model_name + ".gguf"

        self.llm = LlamaCpp(
            model_path=model_path,
            temperature=0.1,
            max_tokens=512,
            top_p=1,
            callback_manager=None,  # callback_manager,
            verbose=True,  # Verbose is required to pass to the callback manager
            n_ctx=2048,
            n_gpu_layers=24,
            stop=['Human', 'Human:', 'bot']
        )

        print("Model Generated")

    def get_model_name(self):
        return self.model_name

    def get_kb_name(self):
        return self.kb_name

    def get_conv_chain(self):

        template = PROMPT_TEMPLATE

        prompt = PromptTemplate(
            input_variables=['input'],
            template=template
        )

        chain = LLMChain(
            llm=self.llm,
            output_parser=StrOutputParser(),
            prompt=prompt,
            verbose=True
        )
        return chain

    def get_kb_chain(self):

        system_message_prompt = SystemMessagePromptTemplate.from_template(PROMPT_TEMPLATE_KB)
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

        chain_from_docs = (
                {
                    "context": lambda input: format_docs(input["documents"]),
                    "question": itemgetter("question"),
                }
                | chat_prompt
                | self.llm
                | StrOutputParser()
        )

        chain_with_source = RunnableParallel(
            {"documents": ensemble_retriever, "question": RunnablePassthrough()}
        ) | {
                    "documents": lambda input: [doc.metadata for doc in input["documents"]],
                    "answer": chain_from_docs,
            }

        return chain_with_source

    # async def model_chat(self):
    #
    #     self.converse = self.get_conv_chain()
    #     print("completed model creation")
    #     return self.converse

    # async def model_kb(self):
    #
    #     self.converse = self.get_kb_chain()
    #     print("completed kb and model creation")
    #     return self.converse

    # def get_model(self):
    #     return self.converse
