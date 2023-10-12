FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt 

COPY ["lgbm_model.pkl", "inorm.py", "./"]

ENTRYPOINT ["gunicorn", "--bind=localhost:8989", "inorm:app"]

