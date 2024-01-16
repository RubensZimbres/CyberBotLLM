from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
from langchain import OpenAI, LLMChain
from langchain.prompts import Prompt
import sys
import subprocess
import os
from langchain.text_splitter import Language
from google.cloud import aiplatform
import vertexai
from vertexai.language_models import CodeGenerationModel
from langchain.llms import VertexAI
from langchain.embeddings import VertexAIEmbeddings
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import RetrievalQA
import os
from langchain.schema.document import Document
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.prompts import QA_PROMPT
from langchain.prompts.chat import SystemMessagePromptTemplate,HumanMessagePromptTemplate
from vertexai.preview.generative_models import GenerativeModel
from vertexai.generative_models._generative_models import HarmCategory, HarmBlockThreshold, ResponseBlockedError
import google.generativeai as genai

def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

def execute_commands(input_string):
    if "$" in input_string:
        commands = input_string.split("$")[1:]  # Split input_string after the "$" symbol

        outputs = []
        for command in commands:
            command = command.strip()  # Remove leading/trailing whitespaces
            output = subprocess.check_output(command, shell=True).decode().strip()
            outputs.append(output)

        return outputs
    else:
        return None

def add_text_to_file(filename, text):
      directory = "training/facts/"
      filepath = directory + filename
      with open(filepath, "a") as file:
          file.write(text + "\n")
      return filepath

def train():

  trainingData = os.listdir("training/facts/")

  embeddings = VertexAIEmbeddings(
      requests_per_minute=5,
      num_instances_per_batch=100,
      model_name = "textembedding-gecko",max_output_tokens=512,
    temperature=0.1,
    top_p=0.8,
    top_k=40
  )

  text_splitter = RecursiveCharacterTextSplitter.from_language(
      language=Language.PYTHON,chunk_size=100, chunk_overlap=0
  )


  docs=[]
  for training in trainingData:
      with open('training/facts/'+training) as f:
          print(f"Add {f.name} to dataset")
          texts=text_splitter.create_documents([f.read()])
          docs+=texts
  
  store = FAISS.from_documents(docs, embeddings)
  store.save_local("faiss_index")
  
  
  
def runPrompt(option):

  history=[]
  
  memory = ConversationBufferMemory(
          memory_key='chat_history', return_messages=True, output_key='answer')


  embeddings = VertexAIEmbeddings(
      requests_per_minute=5,
      num_instances_per_batch=100,
      model_name = "textembedding-gecko",max_output_tokens=512,
    temperature=0.1,
    top_p=0.8,
    top_k=40
  )

  store = FAISS.load_local("faiss_index", embeddings)

  retriever = store.as_retriever(
    search_type="similarity",  # Also test "similarity", "mmr"
    search_kwargs={"k": 1},)

  promptTemplate = ""
      
  if option==2:
      promptTemplate="""" Please answer the user question, using your knowledge base  
  and the chat history: {chat_history}. You can also use the {context}.
  Answer in English language.
  """  
  elif option==3:
      promptTemplate=""""
      You are a security expert and you have to answer to the user that is asking questions.
  Please answer the user question, using your knowledge base  
  and the chat history: {chat_history}, and if you need,  {context}.
  Also, if applicable, identify which cybersecurity concept
  is being neglected and consider information
  in your knowledge base regarding NIST Cybersecurity Framework, CIA tryad, CISSP domains and OSI model
  to support your answer and provide the user with best practices regarding security controls about this
  cybersecurity concept. Answer in english language.
  """
  elif option==4:
      promptTemplate=""""
      You are a security expert and you have to answer to the user that is asking questions {question}.
  Please answer the user question, using your knowledge base  
  and the chat history: {chat_history}. Note that we have user names and passwords in {context}
  Also, if applicable, identify which cybersecurity concept
  is being neglected and consider information
  in your knowledge base regarding NIST Cybersecurity Framework, CIA tryad, CISSP domains and OSI model
  to support your answer and provide the user with best practices regarding security controls about this
  cybersecurity concept. Answer in english language.
  """
  else:
      promptTemplate=""""You are a Google Cloud Security expert and you have to answer to the user that is asking questions.
  Please answer the user question, using your knowledge base  
  and the chat history: {chat_history}, and if you need,  {context}.
  Also, if applicable, identify which cybersecurity concept
  is being neglected and consider information
  in your knowledge base regarding NIST Cybersecurity Framework, CIA tryad, CISSP domains and OSI model
  to support your answer and provide the user with best practices regarding security controls about this
  cybersecurity concept. Answer in english language.
  """

  messages = [
        SystemMessagePromptTemplate.from_template(promptTemplate),
        HumanMessagePromptTemplate.from_template("{question}")
        ]
  qa_prompt = ChatPromptTemplate.from_messages(messages)
  
  llm=VertexAI(model_name="gemini-pro",max_output_tokens=512,temperature=0.2,top_p=1, top_k= 1
    ,safety_settings = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }
)

  qa_chain = ConversationalRetrievalChain.from_llm(
      llm, retriever, memory=memory,get_chat_history=lambda h : h,combine_docs_chain_kwargs={"prompt": qa_prompt})
  

  
  def onMessage(question):
    answer = qa_chain({"question":question,"chat_history":history})#,return_only_outputs=True)
    history.append((question, answer))

    return answer["answer"]
  
  while True:
    prompt_=colored(255, 0, 255,"Ask a question >")
    question = input(prompt_)
    try:
       answer = onMessage(question)
       print('\n',colored(0,255,0,"CyberbotLLM: "),answer,'\n')
    except:
       answer = ""
       print('\n',colored(255,0,0,"CyberbotLLM: "),"Sorry, your question is harmful.",'\n')
       continue
