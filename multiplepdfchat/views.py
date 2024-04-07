from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.response import Response
from rest_framework import status
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import faiss
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# Create your views here.

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = faiss.FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

@api_view(['POST'])
@csrf_exempt
def get_pdf_text(request):
    if request.method == 'POST':
        pdf_files = request.FILES.getlist('pdf_files')
        for pdf_file in pdf_files:
            # Process the PDF file using PyPDF2 or another library
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            # Extract text from the PDF, or perform other operations
            text = ''
            for page in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page].extract_text()
        text_chunks = get_text_chunks(text)
        get_vector_store(text_chunks)
        return HttpResponse('PDF files uploaded successfully')
    else:
        return HttpResponse('Invalid request method')
    
@api_view(['POST'])
@csrf_exempt
def user_input(request):
    if request.method == 'POST':
        user_question = request.data['user_question']
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        new_db = faiss.FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain.invoke(
            {"input_documents":docs, "question": user_question},
             return_only_outputs=True)
        return Response(response, status=status.HTTP_200_OK) 
        