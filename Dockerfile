FROM python:3.11.6-slim-bookworm

WORKDIR /
COPY app ./app/
COPY requirements.txt ./requirements.txt

RUN git clone https://huggingface.co/herrsch99/vegetables
RUN pip install -r requirements.txt
EXPOSE 8084

CMD ["python", "./app/app.py"]