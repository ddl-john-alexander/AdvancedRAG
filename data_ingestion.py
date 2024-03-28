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
import warnings
warnings.filterwarnings('ignore')

class Watcher:
    
    def __init__(self, directory="/mnt/data/AdvancedRAG/", handler=FileSystemEventHandler(), recursive=False):
        self.observer = Observer()
        self.handler = handler
        self.directory = directory
        self.recursive = recursive

    def run(self):
        self.observer.schedule(
            self.handler, self.directory, self.recursive)
        self.observer.start()
        print(f"Watcher Running in {self.directory}")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.observer.stop()
        self.observer.join()
        print("\nWatcher Terminated\n")      

class OnCreated_Handler(FileSystemEventHandler):
    
    def on_created(self, event):
        if event.is_directory:
            return
        filepath = event.src_path
        print(f"file {filepath} created")
        chunk_size=1000
        chunk_overlap=200
        strip_whitespace = True
        separators=["\n\n", "\n", ".", " ", ""] 
        endpoint="embedding-ada-002ja2"
        datasource_name = "mrag-fin-docs-ja"
        pinecone_env = "gcp-starter"
        index_name = "mrag-fin-docs"
        text_field = "text"
        
        try:
            data = self.load_and_split_data(filepath,chunk_size, chunk_overlap, strip_whitespace, separators)
            pinecone = self.init_datasource(datasource_name, pinecone_env)
            embeddings = self.load_embedding_model(endpoint)       
            vectorstore = self.upsert_data_to_vectorstore(embeddings, data, index_name, text_field)  
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise  
      
        print(f"File {filepath} has been added.")

    def load_and_split_data(self,filepath, chunk_size=1000, chunk_overlap=200, strip_whitespace = True, separators=["\n\n", "\n", ".", " ", ""]):
        # Load a file
        print("** load_and_split_data start")
        try:
            loader = PyPDFLoader(filepath)
            data = loader.load_and_split(RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            strip_whitespace=strip_whitespace,
            add_start_index = True,)) 
            print(f"** load_and_split_data {filepath} success")
        except Exception as err:
            print(f"** load_and_split_data: Unexpected {err=}, {type(err)=}")
            raise           
        
        return data 
              
    def load_embedding_model(self,endpoint="embedding-ada-002ja2"):
        print("** load_embedding_model start")
        try:
            embeddings = MlflowEmbeddings(
                target_uri=os.environ["DOMINO_MLFLOW_DEPLOYMENTS"],
                endpoint=endpoint,
                )
            print(f"** load_embedding_model {endpoint} success")
        except Exception as err:
            print(f"** load_embedding_model: Unexpected {err=}, {type(err)=}")
            raise    
        
        return embeddings
   
    def init_datasource(self,datasource_name = "mrag-fin-docs-ja", pinecone_env="gcp-starter"):
                
        conf = DominoPineconeConfiguration(datasource=datasource_name)
        # The pinecone API key should be provided when creating the Domino Data Source and persisted securely.
        # This api_key variable here is only used for satisfying the native pinecone python client initialization where
        # api_key is a mandatory non-empty field.
        api_key = os.environ.get("DOMINO_VECTOR_DB_METADATA", datasource_name)
        print("** init_datasource start")
        try:
            pinecone.init(
                api_key=api_key,
                environment=pinecone_env,
                openapi_config=conf)
            print(f"** init_datasource {datasource_name} success")             
        except Exception as err:
            print(f"** init_datasource: Unexpected {err=}, {type(err)=}")
            raise               
        return pinecone
                   
    def upsert_data_to_vectorstore(self, embeddings, data, index_name = "mrag-fin-docs", text_field = "text"):
        print("** upsert_data_to_vectorstore start")
        try:
            index = pinecone.Index(index_name)
            print(f"*** upsert_data_to_vectorstore: {index_name} created")            
        except Exception as err:
            print(f"*** upsert_data_to_vectorstore - create index: Unexpected {err=}, {type(err)=}")
            raise    
                   
        # switch back to normal index for langchain
        try:
            vectorstore = Pinecone(
                index, embeddings, text_field # Using embedded data from Domino AI Gateway Endpoint
            )
            print(f"*** upsert_data_to_vectorstore: vectorstore created") 
        except Exception as err:
            print(f"*** upsert_data_to_vectorstore - create vectorstore: Unexpected {err=}, {type(err)=}")
            raise            
        
        try:
            docsearch = vectorstore.add_texts([d.page_content for d in data])
            print(f"*** upsert_data_to_vectorstore: vectors upserted to {index_name}") 
     
        except Exception as err:
            print(f"*** upsert_data_to_vectorstore - upsert_data: Unexpected {err=}, {type(err)=}")
            raise
        print(f"** upsert_data_to_vectorstore success")                       
        return vectorstore                      

if __name__ == "__main__":
    directory_to_watch = "/mnt/data/AdvancedRAG/"
    w = Watcher(directory_to_watch, OnCreated_Handler(),recursive=False)
    w.run()