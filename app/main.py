from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import requests
import uuid
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# In memory storing the data  
data_store = {}

# Making model for URL input
class URLRequest(BaseModel):
    url: str

# Installing Endpoint 1: Process Web URL
@app.post("/process_url")
def process_url(request: URLRequest):
    response = requests.get(request.url)
    content = response.text  # Assuming plain text
    chat_id = str(uuid.uuid4())
    data_store[chat_id] = content
    return {"chat_id": chat_id, "message": "URL content processed and stored successfully."}

# Installing Endpoint 2: Process PDF Document
@app.post("/process_pdf")
def process_pdf(file: UploadFile = File(...)):
    pdf_reader = PyPDF2.PdfReader(file.file)
    text = "".join([page.extract_text() for page in pdf_reader.pages])
    chat_id = str(uuid.uuid4())
    data_store[chat_id] = text
    return {"chat_id": chat_id, "message": "PDF content processed and stored successfully."}

# Installing Endpoint 3: Chat with Processed Content
@app.post("/chat")
def chat(chat_id: str, question: str):
    if chat_id not in data_store:
        return {"error": "Invalid chat_id"}
    content = data_store[chat_id]
    documents = [content, question]
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    similarity = cosine_similarity([vectors[1]], [vectors[0]])
    return {"response": f"The main idea of the document is... (similarity: {similarity[0][0]})"}
