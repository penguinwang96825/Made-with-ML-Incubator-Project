FROM fnndsc/ubuntu-python3

WORKDIR /app

ADD . ./

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]

CMD ["app.py"]
