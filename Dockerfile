FROM registry.access.redhat.com/ubi9/python-311

RUN mkdir -p /home/rag 
COPY . /home/rag/
WORKDIR /home/rag

RUN pip install -r requirements.txt
