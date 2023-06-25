from fastapi import FastAPI, Body, UploadFile, File
from fastapi.responses import FileResponse
from core.text_handler import get_result_dict, write_to_excel
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

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
    print('-Started Upload File-')
    df = pd.read_excel(file.file.read())
    file.file.close()
    rd = get_result_dict(df['responsibilities(Должностные обязанности)'].tolist())
    write_to_excel(rd, './database/result.xlsx')



@app.get("/download")
def download_file():
    file_path = './database/result.xlsx'
    return FileResponse(path=file_path, filename=file_path)


@app.post("/text")
def get_input_text(data=Body()):
    print('-Started Upload Text-')
    class_suggestions = get_result_dict([data['text']])
    return class_suggestions
