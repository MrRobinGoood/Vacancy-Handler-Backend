from typing import Union, Annotated
from fastapi import FastAPI
from fastapi import FastAPI, Body, status, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from text_handler import get_result_dict, write_to_excel
import pandas as pd

app = FastAPI()


class Resp:
    def __init__(self, name):
        self.name = name


app = FastAPI()


@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    df = pd.read_excel(file.file.read())
    file.file.close()
    rd = get_result_dict(df['responsibilities(Должностные обязанности)'].tolist())
    write_to_excel(rd)
    file_path = 'resources/result.xlsx'
    return FileResponse(path=file_path, filename=file_path)


@app.post("/text")
def get_input_text(data=Body()):
    print('start')
    class_suggestions = get_result_dict([data['text']])
    print(class_suggestions)
    return class_suggestions
