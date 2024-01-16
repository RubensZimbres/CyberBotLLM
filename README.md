# CyberBotLLM
  
## _Your Google Cloud Cybersecurity Expert, powered by Gemini Generative AI_
  
This chatbot is part of the OWASP educational initiative in Cybersecurity, is based on Google technology, and uses 4 different approaches:
- Regular conversation
- Cybersecurity Expert
- Hardened Cybersecurity Expert
- Cloud Cybersecurity Expert
  
## Features

It uses <b>Langchain</b> to generate a conversation flow with memory, and also a <b>RAG (Retrieval Augmented Generation)</b> document that can be customized according to your educational goals.  
Currently, RAG contains a list of fictitious usernames and passwords. One of the goals is to retrieve usernames and passwords via prompt injection techniques (direct and indirect). You can also poison the RAG document to force a Sensitive Information Disclosure.

## How to use it

In order to use/replicate this chatbot, `git clone` this repository. Then, you will have to create a Google Cloud project, go to IAM, Service Accounts and generate a key.json.  
  
This key can be used directly as an environment variable by using `os.environ['GOOGLE_APPLICATIONS_CREDENTIALS']='key.json'`, or even better, you can go to Google Cloud <b>Secret Manager</b> and create 
a secret called `GOOGLE_APPLICATIONS_CREDENTIALS` and store the content of this JSON file. Main,py file is set up to be used with the Secret Manager in VSCode. 
  
Then, edit your project name and number in main.py. After that, run in command line:  
  
```sh
gcloud auth login
gcloud config set project your project
```

And you are good to go:

```sh
python3 main.py
```
<p align="center">
<img src="https://github.com/RubensZimbres/CyberBotLLM/blob/main/pictures/gemini_0.png">
</p>
