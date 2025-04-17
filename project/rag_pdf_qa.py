import os
import faiss
import gradio as gr
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ========== Step 1: Extract Q&A pairs from PDFs ========== #
def extract_qna_from_pdfs(folder_path):
    qa_pairs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    words = page.extract_words(extra_attrs=["fontname"], use_text_flow=True)

                    question_buffer = []
                    for word in words:
                        text = word['text'].strip()
                        font = word['fontname'].lower()

                        if 'bold' in font:
                            question = " ".join(question_buffer).strip()
                            answer = text
                            if question:
                                qa_pairs.append((question, answer))
                            question_buffer = []  # reset for next question
                        else:
                            question_buffer.append(text)
    return qa_pairs

# ========== Step 2: Build FAISS Index ========== #
def build_faiss_index_from_qna(qa_pairs):
    questions = [q for q, a in qa_pairs]
    answers = [a for q, a in qa_pairs]
    embeddings = embedding_model.encode(questions)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, questions, answers

# ========== Step 3: Keyword Search QA Function ========== #
def create_qa_function(questions, answers):
    def search_keyword(keyword):
        keyword = keyword.lower()
        matched_qna = []

        for q, a in zip(questions, answers):
            if keyword in q.lower():
                matched_qna.append(f"<b>Q:</b> {q}<br><b>A:</b> <b>{a}</b>")

        if not matched_qna:
            return "❌ No relevant Q&A pairs found for the given keyword."

        return "<br><br>".join(matched_qna)

    return search_keyword

# ========== Step 4: Setup ========== #
pdf_folder = "/Users/code/project/your_pdfs_folder"  # <- change this to your actual folder path
print("Extracting Q&A pairs from PDFs...")
qa_pairs = extract_qna_from_pdfs(pdf_folder)
print(f"✅ Extracted {len(qa_pairs)} Q&A pairs.")

faiss_index, questions, answers = build_faiss_index_from_qna(qa_pairs)
qa_func = create_qa_function(questions, answers)

# ========== Step 5: Gradio Interface ========== #
iface = gr.Interface(
    fn=qa_func,
    inputs=gr.Textbox(lines=2, placeholder="Enter keyword (e.g., surplus, price, demand)..."),
    outputs=gr.HTML(),
    title="📘 Smart PDF MCQ Extractor",
    description="Search for questions by keyword. Extracts MCQs from PDF files with bold answers.",
)
iface.launch()
