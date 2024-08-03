# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher
from weather import Weather
import requests as rq


cities = {
    'в москве': 'Moscow,ru',
    'в Москве': 'Moscow,ru',
    'в питере': 'Saint Petersburg,ru',
    'в Питере': 'Saint Petersburg,ru',
    'в Петербурге': 'Saint Petersburg,ru',
    'а в Питере?': 'Saint Petersburg,ru',
    'а в Питере': 'Saint Petersburg,ru',
    'а в питере': 'Saint Petersburg,ru',
}


class ActionWeather(Action):

    def name(self) -> Text:
        return "action_weather_api"

    def run(self, dispatcher: CollectingDispatcher,
    tracker: Tracker,
    domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        city = tracker.latest_message['text']
        weather_data = Weather(cities[city])
        temp = int(weather_data['temp']-273)
        press = int(weather_data['pressure'] * .00750062 * 100)
        dispatcher.utter_message(response="utter_temp", temp=temp, press=press)

        return []


# API LLM block start ###############################################
API_URL_LLM = 'http://127.0.0.1:8000'
def get_model_name():

    try:
        response = rq.get(API_URL_LLM)
    except:
        return 0

    if response.status_code == 200:
        response = response.json()
        return response['message']
    else:
        return 0


def get_knowledge_base_name():

    try:
        response = rq.get(API_URL_LLM + '/getkbname')
    except:
        return 0

    if response.status_code == 200:
        response = response.json()
        return response['message']
    else:
        return 0


def get_model_response(query):

    try:
        response = rq.get(API_URL_LLM + '/q?query=' + query)
    except:
        return 0

    if response.status_code == 200:
        response = response.json()
        return response
    else:
        return 0


def get_kb_response(query):

    try:
        response = rq.get(API_URL_LLM + '/qkb?query=' + query)
    except:
        return 0

    if response.status_code == 200:
        response = response.json()
        return response
    else:
        return 0


def get_rephrase(history, query):

    q_string = f"{API_URL_LLM}/rephrase?history={history}&query={query}"

    try:
        response = rq.get(q_string)
    except:
        return 0

    if response.status_code == 200:
        response = response.json()
        return response
    else:
        return 0


def get_relevant_docs(query):

    try:
        response = rq.get(API_URL_LLM + '/rd?query=' + query)
    except:
        return 0

    if response.status_code == 200:
        response = response.json()
        return response
    else:
        return 0
# API LLM end of block ##############################################


class ActionLLMOn(Action):

    def name(self) -> Text:
        return "action_llm_on"

    def run(self, dispatcher: CollectingDispatcher,
    tracker: Tracker,
    domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        model_name = get_model_name()
        if model_name == 0:
            dispatcher.utter_message(text="Сервис LLM не доступен.")
            return []

        dispatcher.utter_message(text="Подключена модель " + model_name)

        return [SlotSet("llm_on", True)]


class ActionKBOn(Action):

    def name(self) -> Text:
        return "action_kb_on"

    def run(self, dispatcher: CollectingDispatcher,
    tracker: Tracker,
    domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        kb_name = get_knowledge_base_name()
        if kb_name == 0:
            dispatcher.utter_message(text="База знаний не доступна. Увы")
            return []

        dispatcher.utter_message(response="utter_knowledge_base_on", knowledge_base_name=kb_name)

        return [SlotSet("knowledge_base_on", True)]


# clear llm, kb, memory and diagnostic slots
class ActionLLMOff(Action):

    def name(self) -> Text:
        return "action_llm_off"

    def run(self, dispatcher: CollectingDispatcher,
    tracker: Tracker,
    domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text="Отключены:\n \
                                      Лингвистическая модель,\n \
                                      База знаний,\n \
                                      Диагностический режим.\n \
                                      Память очищена")

        return [SlotSet("llm_on", False), SlotSet("knowledge_base_on", False),
                SlotSet("dialog_memory", None), SlotSet("diagnostic_on", False)]


# knowledge base question & answer + formatter
def knowledge_base_q_a(standalone_question, diagnostic_on, memory, dispatcher):

    response = get_kb_response(standalone_question)

    # the llm api not available or response code is not 200
    if response == 0:
        dispatcher.utter_message(text="Что-то пошло не так")
        return []

    full_response = ""
    if response.get('answer', 0) != 0:

        if diagnostic_on:
            kb_name = get_knowledge_base_name()
            dispatcher.utter_message(response='utter_knowledge_base_on', knowledge_base_name=kb_name)

        answer = response['answer']
        sources = response.get('documents', None)
        full_response += answer

        if sources:
            full_response += '\n\n<b>Источники:</b>'

            for n, source in enumerate(sources):
                full_response += f"\n{n + 1}. {'/'.join(list(source.values()))}"

        dispatcher.utter_message(text=full_response)

        print('dialog memory:\n', memory)

        memory = f"HUMAN: {standalone_question}\n AI: {answer}"
        return [SlotSet("dialog_memory", memory)]

    else:
        dispatcher.utter_message(text="Что-то пошло не так. Опять!")
        return []


class ActionLLMQuery(Action):

    def name(self):
        return "action_llm_query"

    def run(self, dispatcher, tracker, domain):

        # slots:
        llm_on = tracker.get_slot("llm_on")
        kb_on = tracker.get_slot("knowledge_base_on")
        memory = tracker.get_slot("dialog_memory")
        diagnostic_on = tracker.get_slot("diagnostic_on")

        text = tracker.latest_message["text"]

        # fallback diagnostic
        latest = tracker.latest_message["intent_ranking"][1]
        intent_name = latest['name']
        confidence = round(float(latest['confidence']), 2)
        print('intent name:', intent_name, 'confidence:', confidence)

        # experimental: kb follow up on fallback
        if kb_on and memory is not None:

            # rephrase with memory
            response = get_rephrase(memory, text)

            if response == 0:
                dispatcher.utter_message(text="Что-то пошло не так")
                return []

            if response.get('text', 0) != 0:
                standalone_question = response['text']

                docs = get_relevant_docs(standalone_question)

                # the retriever did not return relevant documents
                if not docs:
                    dispatcher.utter_message(text="В моей базе знаний нет информации для ответа на данный вопрос")
                    return [SlotSet("dialog_memory", None)]

                return knowledge_base_q_a(standalone_question, diagnostic_on, memory, dispatcher)

        # if llm_on slot did not set ##############################
        if not llm_on:
            dispatcher.utter_message(response="utter_ask_rephrase")
            return []

        # from the list of intents get the second higher predicted intent
        # first will be nlu_fallback
        print(tracker.latest_message["intent_ranking"][0])
        print(tracker.latest_message["intent_ranking"][1])
        print()

        response = get_model_response(text)

        if response == 0:
            dispatcher.utter_message(text="Что-то пошло не так")
            return []

        if response.get('text', 0) != 0:
            response = response['text']
            dispatcher.utter_message(text=response)
        else:
            dispatcher.utter_message(text="Что-то пошло не так. Опять!")
            return []

        return []


# knowledge base q&a with intent
class ActionKBQuery(Action):

    def name(self):
        return "action_kb_query"

    def run(self, dispatcher, tracker, domain):

        kb_on = tracker.get_slot("knowledge_base_on")
        diagnostic_on = tracker.get_slot("diagnostic_on")
        memory = tracker.get_slot("dialog_memory")

        # knowledge base "on" slot is false
        if not kb_on:
            dispatcher.utter_message(response="utter_knowledge_base_off")
            return []

        standalone_question = tracker.latest_message["text"]
        docs = get_relevant_docs(standalone_question)

        # the retriever did not return relevant documents
        if not docs:
            dispatcher.utter_message(text="В моей базе знаний нет информации для ответа на данный вопрос")
            return [SlotSet("dialog_memory", None)]

        return knowledge_base_q_a(standalone_question, diagnostic_on, memory, dispatcher)


class DiagnosticOff(Action):

    def name(self) -> Text:
        return "action_diagnostic_off"

    def run(self, dispatcher: CollectingDispatcher,
    tracker: Tracker,
    domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text="Диагностика отключена")
        return [SlotSet("diagnostic_on", False)]


class DiagnosticOn(Action):

    def name(self) -> Text:
        return "action_diagnostic_on"

    def run(self, dispatcher: CollectingDispatcher,
    tracker: Tracker,
    domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text="Включен диагностический режим")
        return [SlotSet("diagnostic_on", True)]


class ActionDiagnostic(Action):

    def name(self):
        return "action_diagnostic"

    def run(self, dispatcher, tracker, domain):

        diagnostic_on = tracker.get_slot("diagnostic_on")

        if diagnostic_on:
            latest = tracker.latest_message["intent_ranking"][0]
            intent_name = latest['name']
            confidence = round(float(latest['confidence']), 2)
            dispatcher.utter_message(response="utter_diagnostic", intent_name=intent_name, confidence=confidence)
            return []
        else:
            return []

