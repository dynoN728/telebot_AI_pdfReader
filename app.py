import os
from PyPDF2 import PdfReader
import requests
import io
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from google.cloud import texttospeech as tts

import telebot

load_dotenv()

BOT_TOKEN = os.environ.get('BOT_TOKEN')

# Instantiates a client
client = tts.TextToSpeechClient()

backendQuery = "summarize this as best as you can"

bot = telebot.TeleBot(BOT_TOKEN)
print("commit")
knowledgeDB = None #had to declare globally sorry computer

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "upload a pdf file")

@bot.message_handler(content_types=['document'])
def doc_handler(message):
    file_id = message.document.file_id
    file_info = bot.get_file(file_id)

    if file_info is None:
        bot.reply_to(message, "Failed to retrieve file information.")
        return

    pdf = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_info.file_path}"

    try:
        #time to read the pdf
        response = requests.get(pdf)
        Pdf_Reader = PdfReader(io.BytesIO(response.content))
        text = ""
        #initialise an empty string
        #loop thru the whole content and concat into a string of text which is unstrucrued data
        for page in Pdf_Reader.pages:
            text += page.extract_text() #function in-built in pdf-reader library

        #rgiht now i am going to use langchain to split the text with overlaying so that theres greater accuracy
        #example: "i like apples and apples are great however carrots are bad"
        #para 1: "i like apples and apples"
        #para2: "apples and apples are great however"
        #para3: "great however carrots are bad"
        #this ensures that the data gets a greater accuracy 

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=2500, # takes up to 2500 words before splitting it next
            chunk_overlap=100, # this was the example i was talking about
            length_function=len # basic python length function
        )

        chunks = text_splitter.split_text(text)

    except Exception as e:
        bot.reply_to(message, f"An error occurred: {str(e)}")
        return  # Exit the function if an error occurs

    # Send each chunk as a separate message
    for chunk in chunks:
        bot.reply_to(message, chunk)

    # Create embeddings 
    embeddings = OpenAIEmbeddings() 

    # Seacrching time using FAISS (facebook ai similarity seacrch)
    global knowledgeDB
    knowledgeDB = FAISS.from_texts(chunks, embeddings)

    if knowledgeDB is not None:
        docs = knowledgeDB.similarity_search(backendQuery)

        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=backendQuery)
        bot.reply_to(message, response)

        #run tts

        
    else:
        bot.reply_to(message, "please give a pdf file first")

# We get the user to ask questions about the pdf now

# @bot.message_handler(func=lambda message: True)
# def reply_to_question(message):
#     query = message.text
#     if not query.startswith("/"):  
#         if knowledgeDB is not None:
#             docs = knowledgeDB.similarity_search(query)

#             llm = OpenAI()
#             chain = load_qa_chain(llm, chain_type="stuff")
#             response = chain.run(input_documents=docs, question=query)
#             bot.reply_to(message, response)
#         else:
#             bot.reply_to(message, "please give a pdf file first")
    

bot.infinity_polling() # FOC taught this polling and interrupt
