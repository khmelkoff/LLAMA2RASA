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


class ActionLLMOff(Action):

    def name(self) -> Text:
        return "action_llm_off"

    def run(self, dispatcher: CollectingDispatcher,
    tracker: Tracker,
    domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text="Лингвистическая модель отключена")

        return [SlotSet("llm_on", False)]


class ActionLLMQuery(Action):

    def name(self):
        return "action_llm_query"

    def run(self, dispatcher, tracker, domain):
        llm_on = tracker.get_slot("llm_on")
        latest = tracker.latest_message["intent_ranking"][1]
        intent_name = latest['name']
        confidence = round(float(latest['confidence']), 2)

        if not llm_on:
            dispatcher.utter_message(response="utter_ask_rephrase")
            dispatcher.utter_message(response="utter_diagnostic", intent_name=intent_name, confidence=confidence)
            return []

        # from the list of intents get the second higher predicted intent
        # first will be nlu_fallback
        print(tracker.latest_message["intent_ranking"][0])
        print(tracker.latest_message["intent_ranking"][1])
        print()

        text = tracker.latest_message["text"]
        response = get_model_response(text)

        if response == 0:
            dispatcher.utter_message(text="Что-то пошло не так.")
            return []

        if response.get('text', 0) != 0:
            response = response['text']
            dispatcher.utter_message(text=response)
        else:
            dispatcher.utter_message(text="Что-то пошло не так. Опять!")
            return []

        return []


class ActionKBQuery(Action):

    def name(self):
        return "action_kb_query"

    def run(self, dispatcher, tracker, domain):

        llm_on = tracker.get_slot("llm_on")

        if not llm_on:
            dispatcher.utter_message(response="utter_knowledge_base_off")
            return []

        text = tracker.latest_message["text"]
        response = get_kb_response(text)

        if response == 0:
            dispatcher.utter_message(text="Что-то пошло не так.")
            return []

        full_response = ""

        if response.get('answer', 0) != 0:

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

        else:
            dispatcher.utter_message(text="Что-то пошло не так. Опять!")

        return []


class ActionDiagnostic(Action):

    def name(self):
        return "action_diagnostic"

    def run(self, dispatcher, tracker, domain):

        latest = tracker.latest_message["intent_ranking"][0]
        intent_name = latest['name']
        confidence = round(float(latest['confidence']), 2)
        dispatcher.utter_message(response="utter_diagnostic", intent_name=intent_name, confidence=confidence)
        return []