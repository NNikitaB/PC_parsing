FROM python:3.10

WORKDIR /code
COPY ./requirements.txt .
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY . .
EXPOSE 8025
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8025"]
#ENTRYPOINT ./run.sh