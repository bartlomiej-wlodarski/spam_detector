FROM python:3.9
ADD . /python-flask
WORKDIR /python-flask
RUN pip install -r requirements.txt
RUN python3 -m nltk.downloader stopwords
RUN python3 -m nltk.downloader wordnet
RUN python3 -m nltk.downloader punkt
RUN python3 -m nltk.downloader omw-1.4