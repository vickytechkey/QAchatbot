FROM python:3.10.11

WORKDIR /app

COPY requirements.txt .
COPY qachatbot .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:80"]
