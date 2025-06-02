# 📄 MCQ PDF Keyword Q&A Search

A Python tool to extract multiple-choice questions (MCQs) and answers from PDFs where answers are in **bold text**, and build a keyword-based searchable Q&A interface using Gradio.

---

## Features

- Extract question and answer pairs from PDF files.
- Detect answers based on bold font style in the PDFs.
- Use Sentence Transformers to embed questions.
- Build a FAISS index for fast similarity search.
- Provide a user-friendly web interface for keyword search using Gradio.
- Return matching questions and answers with answers highlighted in bold.

---

Install dependencies:
pip install -r requirements.txt

Add your PDF files containing MCQs to the pdfs folder (create the folder if it doesn't exist)

Run the main script:

python your_script_name.py

The script will:
Extract Q&A pairs from PDFs in the pdfs folder.
Build a FAISS index for question embeddings.
Launch a Gradio web interface for keyword-based Q&A search.
Open the URL printed in your terminal (typically http://127.0.0.1:7860) in your browser to use the interface.

How It Works
PDF Parsing:
Reads PDF pages using pdfplumber, extracts words with font info, and identifies answers by detecting bold fonts.
Embedding:
Encodes questions into dense vectors using the all-MiniLM-L6-v2 model from sentence-transformers.
Indexing:
Builds a FAISS index for fast similarity search (note: the current keyword search uses substring matching).
Search Interface:
Provides a Gradio UI where users input keywords to retrieve related Q&A pairs.

Dependencies
pdfplumber
sentence-transformers
faiss-cpu
gradio
numpy


Install all dependencies with:
pip install pdfplumber sentence-transformers faiss-cpu gradio numpy

Example Output
🔍 Extracting Q&A pairs from PDFs...
✅ Extracted 120 Q&A pairs.

1. Q: What is the capital of France?
   A: Paris

2. Q: Which element has the atomic number 6?
   A: Carbon

License
This project is licensed under the MIT License. See the LICENSE file for details.



