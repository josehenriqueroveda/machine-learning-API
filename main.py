from typing import List
from fastapi import FastAPI
from ml import nlp
from pydantic import BaseModel
import starlette


app = FastAPI()


@app.get('/')
def read_main():
    return {'message': 'Hello World'}


class Article(BaseModel):
    content: str
    comments: List[str] = []


@app.post('/article/')
def analyze_article(articles: List[Article]):
    ents = []
    comments = []
    for article in articles:
        for comment in comments:
            comments.append(comment.upper())
        doc = nlp(article.content)
        for ent in doc.ents:
            ents.append({'text': ent.text, 'label': ent.label_})
    return {'ents': ents, 'comments': comments}
