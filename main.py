import os
import sys
import select
import time
import subprocess
import warnings
from google.cloud import aiplatform
import vertexai
from vertexai.language_models import CodeGenerationModel
from google.cloud import secretmanager
warnings.filterwarnings('ignore')

def access_secret_version(secret_version_id):
  client = secretmanager.SecretManagerServiceClient()
  response = client.access_secret_version(name=secret_version_id)
  return response.payload.data.decode('UTF-8')

secret_version_id = f"projects/123456787654/secrets/GOOGLE_APPLICATION_CREDENTIALS/versions/latest"

key=access_secret_version(secret_version_id)
os.getenv(key)

vertexai.init(project='your-project', location='us-central1')

def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

os.system('clear')
print(colored(0,255,0,"""
              ;           
              0           
             kMk          
           ;XMMMX,        
        .oXMMMMMMMXl.     
   ..:xWMMMMMMMMMMMMMNx:..
        .oXMMMMMMMXl.     
           ;XMMMX,        
             kMk          
              0           
              |         """.replace(',','o')))
if len(key)<1:
  print("You must set an GOOGLE_CLOUD_KEY using the Secrets tool",
        file=sys.stderr)
else:

  print(colored(0,255,0,"[+] == GEMINI CYBER BOT == [+]"),'\n')
  print("You have twelve seconds to select an option:")
  print()
  print("1: Train Model\n2: Talk to your Bot\n3: Talk to your Cybersecurity Expert Bot\n4: Talk to your Hardened Cybersecurity Expert Bot\n5: Talk to your Cloud Cybersecurity Expert Bot>",end="")

  i, o, e = select.select([sys.stdin], [], [], 12)
  print()

  if (i):
    choice = sys.stdin.readline().strip()
    time.sleep(0.5)
    os.system('clear')
    if choice == "1":
      print(colored(0,255,0,"[+] == BOT TRAINING MODE == [+]"),'\n')
      #import get_code
      #get_code.generate_code()
      import process
      process.train()
      print("\nTraining . . .")
      print("RAG "+ colored(0,255,0,"trained")+" =)")
    elif choice == "2":
      print(colored(255,255,0,"[+] == BOT CONVERSATION MODE == [+]\nLoading . . ."),'\n')
      import process
      process.runPrompt(2)
    elif choice == "3":
      print(colored(255,0,0,"[+] == BOT CYBERSECURITY EXPERT == [+]\nLoading . . ."),'\n')
      import process
      process.runPrompt(3)
    elif choice == "4":
      print(colored(255,0,0,"[+] == BOT HARDENED CYBERSECURITY EXPERT == [+]\nLoading . . ."),'\n')
      import process
      process.runPrompt(4)
    else:
      print(colored(0,0,255,"[+] == BOT CLOUD CYBERSECURITY EXPERT == [+]\nLoading . . ."),'\n')
      import process
      process.runPrompt(5)
  else:
    pass
