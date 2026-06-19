# 🧠 ScholarAI – AI Research Assistant

ScholarAI is an AI-powered research paper analysis platform that helps students, researchers, and learners quickly understand academic papers using Google Gemini AI.

The application can summarize research papers, explain concepts in simple language, generate quizzes and interview questions, and even allow users to chat with the paper using Retrieval-Augmented Generation (RAG).

---

## 🚀 Features

### 📄 Research Paper Analysis

- Upload research papers in PDF format
- Automatic text extraction
- Executive summary generation
- Key takeaways extraction
- Difficulty rating (1–10)
- Beginner-friendly explanation (ELI15)

### 🎓 Learning Assistant

- Generate quiz questions and answers
- Generate technical interview questions
- Generate future research ideas

### 🤖 RAG-Powered Chat

- Ask questions directly about the paper
- Semantic search using FAISS vector database
- Context-aware responses
- Source chunk display for transparency

### 💡 Smart User Experience

- Suggested question buttons
- Download analysis as a text file
- Progress indicators during processing
- Large PDF support
- Clear chat functionality
- Interactive Streamlit interface

---

## 🏗️ System Architecture

PDF Upload

↓

PDF Text Extraction

↓

Gemini Analysis Engine

↓

Text Chunking

↓

Gemini Embeddings

↓

FAISS Vector Database

↓

RAG Chat Interface

↓

Question Answering

---

## 🛠️ Tech Stack

### Frontend

- Streamlit

### Backend

- Python

### AI Models

- Google Gemini 2.5 Flash
- Gemini Embedding Model

### AI Frameworks

- LangChain

### Vector Database

- FAISS

### Additional Libraries

- PyPDF
- Python Dotenv

---

##  📸 Screenshots

### Home Page

Home Page

### Analysis Dashboard

Analysis Dashboard

### RAG Chat Assistant

RAG Chat Assistant

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/ScholarAI.git
cd ScholarAI

```

Install dependencies:

```bash
pip install -r requirements.txt

```

Create a `.env` file:

```env
GEMINI_API_KEY=YOUR_API_KEY

```

Run the application:

```bash
streamlit run app.py

```

---

## 📋 Usage

1. Upload a research paper PDF.
2. Click **Analyze Paper**.
3. Review the generated insights.
4. Download the analysis if desired.
5. Ask questions about the paper in the chat interface.
6. Explore source chunks used for answers.

---

## 🎯 Future Improvements

- Multi-paper comparison
- Research paper recommendations
- Citation generation
- Export to PDF
- Research trend analysis
- Academic knowledge graphs

---

## 👩‍💻 Author

**Architha R**

[B.Tech](http://B.Tech) Electrical and Computer Engineering

Interested in:

- Artificial Intelligence
- Machine Learning
- Software Development
- Research Automation

---

## 📜 License

This project is licensed under the MIT License.