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


@app.get("/")
async def root():
    model = model_obj.get_model_name()
    return {"message": model}


@app.get("/getkbname")
async def root():
    kb_name = model_obj.get_kb_name()
    return {"message": kb_name}


# @app.get("/genmodel")
# async def gen_model():
#     await model_obj.model_chat()
#     return {"message": "Model Generated"}


# @app.get("/genkb")
# async def gen_kb():
#     await model_obj.model_kb()
#     return {"message": "KB Generated"}


@app.get("/q")
async def search(query: str):
    print(f"query: {query}")
    # converse = model_obj.get_model()
    converse = model_obj.get_conv_chain()

    if converse is None:
        raise HTTPException(status_code=500, detail="Model Not Generated")

    response = converse.invoke(query)
    print(response)

    return response


@app.get("/qkb")
async def search_kb(query: str):
    print(f"query: {query}")
    # converse = model_obj.get_model()
    converse = model_obj.get_kb_chain()

    if converse is None:
        raise HTTPException(status_code=500, detail="Model Not Generated")

    response = converse.invoke(query)
    print(response)

    return response