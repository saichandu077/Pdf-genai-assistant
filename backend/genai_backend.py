from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv
import pdfplumber
import pytesseract
from PIL import Image
import os
import traceback
import tempfile
import shutil

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


@app.get("/")
def root():
    return {"message": "✅ GenAI Backend is Running"}


@app.post("/ask")
async def ask_question(file: UploadFile = None, question: str = Form(...)):
    try:
        # Case 1: No file, treat as pure ChatGPT
        if not file:
            response = llm.invoke(question)
            return {"answer": response.content}

        text = ""

        if file.filename.endswith(".pdf"):
            try:
                reader = PdfReader(file.file)
                if reader.is_encrypted:
                    reader.decrypt("")
                text = "\n".join([p.extract_text() or "" for p in reader.pages])
            except Exception:
                file.file.seek(0)
                with pdfplumber.open(file.file) as pdf:
                    text = "\n".join([page.extract_text() or "" for page in pdf.pages])

            if not text.strip():
                file.file.seek(0)
                with pdfplumber.open(file.file) as pdf:
                    for page in pdf.pages:
                        img = page.to_image(resolution=300).original
                        text += pytesseract.image_to_string(img)

        elif file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image = Image.open(file.file)
            text = pytesseract.image_to_string(image)

        elif file.filename.endswith(".txt"):
            text = (await file.read()).decode("utf-8")

        else:
            return JSONResponse(status_code=400, content={"error": "❌ Unsupported file type"})

        if not text.strip():
            return JSONResponse(status_code=400, content={"error": "⚠️ No extractable text found."})

        splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(text)
        documents = [Document(page_content=chunk) for chunk in chunks]

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents, embedding=embeddings)

        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
        result = qa_chain.run(question)

        return {"answer": result}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"❌ Internal error: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("genai_backend:app", host="0.0.0.0", port=8000)
