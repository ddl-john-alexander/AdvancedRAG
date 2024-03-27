from domino_data.vectordb import DominoPineconeConfiguration
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import MlflowEmbeddings
from langchain.vectorstores import Pinecone

import csv
import os
import random
import pinecone
import sys

import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from mlflow.deployments import get_deploy_client
import os

client = get_deploy_client(os.environ['DOMINO_MLFLOW_DEPLOYMENTS'])

texts = []
metadata = []
chunk_size=1000
chunk_overlap=200
strip_whitespace = True
separators=["\n\n", "\n", ".", " ", ""]
PINECONE_ENV="gcp-starter"

def load_embedding_model():
    embeddings = MlflowEmbeddings(
        target_uri=os.environ["DOMINO_MLFLOW_DEPLOYMENTS"],
        endpoint="embedding-ada-002ja2",
        )
    return embeddings

def init_datasource():
    datasource_name = "mrag-fin-docs-ja"
    conf = DominoPineconeConfiguration(datasource=datasource_name)
    # The pinecone API key should be provided when creating the Domino Data Source and persisted securely.
    # This api_key variable here is only used for satisfying the native pinecone python client initialization where
    # api_key is a mandatory non-empty field.
    api_key = os.environ.get("DOMINO_VECTOR_DB_METADATA", datasource_name)

    pinecone.init(
        api_key=api_key,
        environment=PINECONE_ENV,
        openapi_config=conf)

    # Previously created index
    index_name = "mrag-fin-docs"
    #index = pinecone.Index(index_name)
    return pinecone

class MyHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        filepath = event.src_path
      
        # Load an entire folder
        loader = PyPDFLoader(filepath)
        data = loader.load_and_split(RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        strip_whitespace=strip_whitespace,
        add_start_index = True,))
        
        pinecone = init_datasource()
        embeddings = load_embedding_model()        

        index_name = "mrag-fin-docs"
        index = pinecone.Index(index_name)
        text_field = "text"
        # switch back to normal index for langchain
        vectorstore = Pinecone(
            index, embeddings, text_field # Using embedded data from Domino AI Gateway Endpoint
        )
          
        docsearch = vectorstore.add_texts([d.page_content for d in data])
        print(f"File {filepath} has been added.")
        
def watch_directory(directory):
    event_handler = MyHandler()
    observer = Observer()
    observer.schedule(event_handler, directory, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    directory_to_watch = "/mnt/data/AdvancedRAG/"
    watch_directory(directory_to_watch)