FROM registry.access.redhat.com/ubi9/python-311

USER 0
RUN cd /home
RUN mkdir rag
COPY . /home/rag/
WORKDIR /home/rag
RUN chmod -R 777 /home/rag/

RUN pip install -r requirements.txt

ENTRYPOINT ["sleep", "infinity"]
