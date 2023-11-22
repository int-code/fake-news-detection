from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from main import check

app = FastAPI()

@app.get("/")
def helloWorld() -> str:
    return "Hello World"

@app.get("/{name}")
def helloName(name:str)-> str:
    return f"Hello, {name}"

@app.get("/get_square/{number}")
def get_square(number:int) -> int:
    return number*number


class NewsData(BaseModel):
    title: str = ''
    text: str = ''


@app.post("/get-news")
def get_news(news_data: NewsData):
    result = check({
        'text':news_data.text,
        'title': news_data.title
    })
    return {
        "result":result.tolist()
    }
