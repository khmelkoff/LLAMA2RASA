from fastapi import FastAPI  # , File, UploadFile, Form
# from pydantic import BaseModel
# import asyncio

from fastapi import Depends, HTTPException

from model import Model
# from dotenv import load_dotenv
# import os
# from typing import List, Annotated

app = FastAPI()
model_obj = Model()
# chat_history = []

# load_dotenv()

# ALLOWED_PDF_EXTENSIONS = [".pdf"]
#
#
# def allowed_file(filename: str) -> bool:
#     ext = os.path.splitext(filename)[1]
#     return ext.lower() in ALLOWED_PDF_EXTENSIONS
#
#
# class Search(BaseModel):
#     query: str


@app.get("/")
async def root():
    model = model_obj.get_model_name()
    return {"message": model}


@app.post("/genmodel")
# async def gen_model(pdf_docs: Annotated[List[UploadFile], File()] = None,
#                     url: Annotated[str, Form()] = None,
#                     ):
async def gen_model():
    # if pdf_docs is None and url is None:
    #     raise HTTPException(status_code=500, detail="Atleast one param of `pdf_docs` and `url` required.")
    #
    # print("uploaded")
    # if pdf_docs is not None:
    #     for file in pdf_docs:
    #         if not allowed_file(file.filename):
    #             raise HTTPException(status_code=400, detail="Invalid file extension")
    # await model_obj.model_data(pdf_docs=pdf_docs, url=url)
    await model_obj.model_data()
    return {"message": "Model Generated"}


@app.get("/search")
async def search(query: str):
    print(f"query: {query}")
    converse = model_obj.get_model()
    if converse is None:
        raise HTTPException(status_code=500, detail="Model Not Generated")

    # response = converse({"question": query})
    # chat_history.append(response['chat_history'])  # TODO

    response = converse.invoke(query)
    print(response)

    return response
