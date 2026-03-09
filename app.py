from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import sqlite3

# Load the API key
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("Missing GROQ_API_KEY. Please check your .env file.")

app = FastAPI(title="AI Student Toolkit API")

llm = ChatGroq(
    temperature=0.7, 
    groq_api_key=api_key, 
    model_name="llama-3.3-70b-versatile" 
)

# --- NEW: Database Setup ---
def setup_database():
    # This creates a file called 'student_data.db' to store our resumes
    conn = sqlite3.connect("student_data.db")
    cursor = conn.cursor()
    # Create a table if it doesn't exist yet
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT,
            resume_text TEXT,
            photo_data TEXT
        )
    ''')
    conn.commit()
    conn.close()

setup_database()

# --- Data Models ---
class PromptRequest(BaseModel):
    tool: str
    text: str

# New model for saving the resume
class SaveResumeRequest(BaseModel):
    student_name: str
    resume_text: str
    photo_data: str 

# --- Endpoints ---
@app.post("/generate")
def generate_ai_response(req: PromptRequest):
    prompts = {
        "chat": "You are a helpful AI software developer assistant. Answer clearly and concisely.",
        "resume": "You are an expert technical recruiter. Convert the following input into a professional, ATS-optimized resume summary and bullet points.",
        "career": "You are a career counselor for computer science students. Suggest 3 tailored career paths based on the input.",
        "interview": "You are a senior technical interviewer. Generate 3 challenging interview questions and answers based on the topic."
    }
    
    sys_prompt = prompts.get(req.tool, prompts["chat"])
    prompt_template = ChatPromptTemplate.from_messages([("system", sys_prompt), ("human", "{text}")])
    chain = prompt_template | llm
    
    try:
        response = chain.invoke({"text": req.text})
        return {"response": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# NEW: Endpoint to save a resume to the database
@app.post("/save_resume")
def save_resume(data: SaveResumeRequest):
    try:
        conn = sqlite3.connect("student_data.db")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO resumes (student_name, resume_text, photo_data) VALUES (?, ?, ?)",
                       (data.student_name, data.resume_text, data.photo_data))
        conn.commit()
        conn.close()
        return {"message": "Resume saved successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database Error: {str(e)}")

# NEW: Endpoint to get all saved resumes
@app.get("/get_resumes")
def get_resumes():
    try:
        conn = sqlite3.connect("student_data.db")
        cursor = conn.cursor()
        cursor.execute("SELECT student_name, resume_text, photo_data FROM resumes")
        rows = cursor.fetchall()
        conn.close()
        
        # Package the data into a list of dictionaries
        resumes = [{"name": row[0], "text": row[1], "photo": row[2]} for row in rows]
        return resumes
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database Error: {str(e)}")