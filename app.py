from fastapi import FastAPI, Body, UploadFile, File
from fastapi.responses import FileResponse
from core.text_handler import get_result_dict, write_to_excel
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


class Resp:
    def __init__(self, name):
        self.name = name


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    print('started upload')
    df = pd.read_excel(file.file.read())
    file.file.close()
    rd = get_result_dict(df['responsibilities(Должностные обязанности)'].tolist())
    print(rd)
    write_to_excel(rd,'./database/result.xlsx')
    # file_path = 'resources/result.xlsx'
    # return FileResponse(path=file_path, filename=file_path)


@app.get("/download")
def download_file():
    file_path = './database/result.xlsx'
    return FileResponse(path=file_path, filename=file_path)


@app.post("/text")
def get_input_text(data=Body()):
    print('start')
    class_suggestions = get_result_dict([data['text']])
    print(class_suggestions)
    return class_suggestions
