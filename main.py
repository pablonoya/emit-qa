from fastapi import FastAPI
from pydantic import BaseModel

from langchain.chains.question_answering import load_qa_chain
from langchain.chains import AnalyzeDocumentChain
from langchain.llms import OpenAI
from PyPDF2 import PdfReader


reader = PdfReader("25 preguntas.pdf")


raw_text = ""
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text

llm = OpenAI(temperature=0)
qa_chain = load_qa_chain(llm, chain_type="map_reduce")

qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)

app = FastAPI()


class Question(BaseModel):
    text: str


@app.post("/question/")
async def question(question: Question):
    answer = qa_document_chain.run(input_document=raw_text, question=question.text)
    return answer.strip()
