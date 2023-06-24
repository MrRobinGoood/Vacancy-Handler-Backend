
FROM python:3.10.9

WORKDIR /backend

COPY requirements.txt /backend/requirements.txt
COPY core /backend/core
COPY database /backend/database
COPY resources /backend/resources
COPY app.py /backend/app.py

RUN pip install --no-cache-dir --upgrade -r /backend/requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]