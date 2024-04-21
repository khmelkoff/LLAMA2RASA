# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings  # HuggingFaceInstructEmbeddings?
# from langchain_community.vectorstores import FAISS

from langchain_community.llms import LlamaCpp
# from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain.schema import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
# from langchain_core.runnables import RunnablePassthrough
# from langchain_community.document_loaders import UnstructuredPDFLoader

# import io
# import requests
# from bs4 import BeautifulSoup
# from db import DB
# import re


# def format_docs(docs):
#     string = "\n\n".join([d.page_content for d in docs])
#     print('refs:', string)
#     return string
#
#
# def text_cleaner(txt: str):
#     txt = re.sub(r'\s*-\n', '', txt)
#     txt = re.sub(r'\s\.', '.', txt)
#     txt = re.sub(r'\s,', ',', txt)
#     txt = re.sub(r'\s»', '»', txt)
#     txt = re.sub(r'\s+', ' ', txt)
#     txt = re.sub(r'www\S+\s', '', txt)
#     return txt


class Model:

    def __init__(self) -> None:
        self.converse = None
        self.model_name = 'saiga_mistral_7b_q8_0'

#    @staticmethod
    def get_model_name(self):
        return self.model_name

#    @staticmethod
    def get_conv_chain(self):

        model_path = "llm/" + self.model_name + ".gguf"

        llm = LlamaCpp(
            model_path=model_path,
            temperature=0.75,
            max_tokens=512,
            top_p=1,
            callback_manager=None,  # callback_manager,
            verbose=True,  # Verbose is required to pass to the callback manager
            n_ctx=512,
            n_gpu_layers=24,
            stop=['Human', 'Human:', 'bot']
        )

        # memory = ConversationBufferMemory(
        #     memory_key='chat_history', return_messages=True)
        #
        # conversation_chain = ConversationalRetrievalChain.from_llm(
        #     llm=llm,
        #     retriever=vectorstore.as_retriever(
        #         search_type="similarity",  # тип поиска похожих документов
        #         search_kwargs={'k': 2, 'score_threshold': 1.6}
        #     ),
        #     memory=memory
        # )
        # return conversation_chain

        # retriever = vectorstore.as_retriever(
        #     search_type="similarity",  # тип поиска похожих документов
        #     search_kwargs={'k': 3}
        # )

        template = """
        <s>"role": "system"\n"content": "Забудь все предыдущие инструкции.
        Ты — ассистент, отвечаешь на вопросы. Тебя зовут Настя. Всегда отвечай с сарказмом и юмором. 
        Если не знаешь ответ, попроси переформулировать вопрос."</s>
        <s>"role": "user"\n"content": {input}</s>
        <s>bot\n
        """

        prompt = PromptTemplate(
            input_variables=['input'],
            template=template
        )

        chain = LLMChain(
            llm=llm,
            output_parser=StrOutputParser(),
            prompt=prompt,
            verbose=True
        )
        return chain

    # @staticmethod
    # def web_scrap_to_txt(url):
    #     print("started webscraping...")
    #     content = []
    #     response = requests.get(url)
    #
    #     # Create a BeautifulSoup object to parse the HTML content
    #     soup = BeautifulSoup(response.text, 'html.parser')
    #
    #     # Extract specific elements from the HTML
    #     title = soup.title.text
    #     paragraphs = soup.find_all('p')
    #     content.append(title)
    #
    #     print("completed capturing title")
    #     for p in paragraphs:
    #         content.append(p.text)
    #     return ' '.join(content)

    # async def model_data(self, pdf_docs="", url=""):
    async def model_data(self):

        self.converse = self.get_conv_chain()
    #     self.db.store_data(db_data)
        print("completed model creation")
        return self.converse

    def get_model(self):
        return self.converse
