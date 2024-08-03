from langchain_community.llms import LlamaCpp
from langchain.schema import StrOutputParser
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from typing import List, Tuple, Final
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Chroma
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
PROMPT_TEMPLATE_EXPERIMENTAL = """
<s>"role": "system"\n"content": "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language, that can be used to query a FAISS index. This query will be used to retrieve documents with additional context. 
Let me share a couple examples that will be important. If this is the second question onwards, you should properly rephrase the question like this: 

Chat History:
Human: Чем кредитная карта отличается от дебетовой?
AI: Кредитная карта - это платежная карта, которая предоставляет вам возможность заемных средств, которые вам необходимо вернуть в срок, установленный банком, а также оплатить проценты за использование средств. Дебетовая карта - это платежная карта, на которой лежат ваши деньги, которые вы можете использовать как угодно, без заемных средств и процентов.
Follow Up Input: Как её оформить?
Standalone Question: Как оформить кредитную карту?

Chat History:
Human: Что такое дебетовая карта?
AI: Дебетовая карта - это платежная карта, на которой лежат ваши деньги.
Follow Up Input: Сколько стоит её обслуживание?
Standalone Question: Сколько стоит обслуживание дебетовой карты?

Now, with those examples, here is the actual chat history and input question.

Chat History: {context}"</s>

<s>"role": "user"\n"content": Follow Up Input: {question}</s>
<s>bot\n Standalone question:
"""

THRESHOLD = 0.25
RELEVANT_K = 3


# Text loader and title splitter
def load_and_split_markdown(file_path: str, splitter: List[Tuple[str, str]]):
    loader = TextLoader(file_path, encoding='utf8')
    docs = loader.load()

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=splitter)
    md_header_splits = markdown_splitter.split_text(docs[0].page_content)
    return md_header_splits

# Retrieved documents formatter
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Embeddings for vector search
embedding = HuggingFaceEmbeddings(
    model_name="cointegrated/LaBSE-en-ru", model_kwargs={"device": "cuda"}
)

docs = load_and_split_markdown(KB_FILE_PATH, HEADERS_TO_SPLIT)
print('Knowledge base loaded, kb chancks:', len(docs))

# DB for our vectors
vectorstore = Chroma.from_documents(documents=docs, embedding=embedding)
print('Vectorstore created')

# Create retriever
retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold", search_kwargs={'k': RELEVANT_K, 'score_threshold': THRESHOLD}
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
            temperature=0.,
            max_tokens=512,
            top_p=1,
            callback_manager=None,  # callback_manager,
            verbose=True,  # Verbose is required to pass to the callback manager
            n_ctx=2048,
            n_gpu_layers=30,
            stop=['Human', 'Human:', 'bot']
        )

        print("Model Generated")

    def get_model_name(self):
        return self.model_name

    def get_kb_name(self):
        return self.kb_name

    @staticmethod
    def get_relevant_docs(query: str):
        return vectorstore.similarity_search_with_relevance_scores(query, k=RELEVANT_K, score_threshold=THRESHOLD)

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

    def get_conv_chain_rephrase(self):

        template = PROMPT_TEMPLATE_EXPERIMENTAL

        prompt = PromptTemplate(
            input_variables=['context', 'question'],
            template=template
        )

        print(prompt)

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
            {"documents": retriever, "question": RunnablePassthrough()}
        ) | {
                    "documents": lambda input: [doc.metadata for doc in input["documents"]],
                    "answer": chain_from_docs,
            }

        return chain_with_source
