import os
import time
import shutil
import threading
import logging
import json
import subprocess
import platform
import sys
import datetime
import customtkinter as ctk
from tkinter import filedialog, messagebox
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import pystray
from PIL import Image, ImageDraw
import requests

# --- LIBRARIES ---
from pypdf import PdfReader
import pandas as pd
try: import docx
except ImportError: docx = None
try: import openpyxl
except ImportError: openpyxl = None

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- CONFIG ---
APP_NAME = "Trove"
VERSION = "v28.0"
CONFIG_FILE = os.path.expanduser("~/smartsort_config.json")
CHAT_HISTORY_FILE = os.path.expanduser("~/smartsort_chat_history.json")
LOG_FILE = os.path.expanduser("~/smartsort_debug.log")
DB_DIR = os.path.expanduser("~/Documents/SmartSort_Local_DB")
TARGET_DIR = os.path.expanduser("~/Documents/SmartSort_Vault")

LOCAL_MODEL = "llama3.2" 

DEFAULT_CONFIG = {
    "target_dir": TARGET_DIR,
    "deep_scan": True,
    "startup_cleanup": True,
    "ai_renaming": True,
    "rag_enabled": True,
    "semantic_rules": {
        "invoice": "Financial/Invoices",
        "receipt": "Financial/Receipts",
        "resume": "HR/Resumes",
        "report": "Work/Reports",
        "contract": "Legal/Contracts"
    },
    "extension_rules": {
        "Images": [".jpg", ".jpeg", ".png", ".gif"],
        "Documents": [".pdf", ".docx", ".txt", ".xlsx", ".csv"],
        "Archives": [".zip", ".rar"],
        "Installers": [".dmg", ".pkg", ".exe"]
    }
}

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(message)s')

def load_config():
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'w') as f: json.dump(DEFAULT_CONFIG, f, indent=4)
    try:
        with open(CONFIG_FILE, 'r') as f: return json.load(f)
    except: return DEFAULT_CONFIG

def save_config(new_config):
    with open(CONFIG_FILE, 'w') as f: json.dump(new_config, f, indent=4)

class ChatManager:
    @staticmethod
    def load_chats():
        if not os.path.exists(CHAT_HISTORY_FILE): return {}
        try:
            with open(CHAT_HISTORY_FILE, 'r') as f: return json.load(f)
        except: return {}

    @staticmethod
    def save_chat(session_id, history):
        chats = ChatManager.load_chats()
        chats[session_id] = history[-20:] 
        with open(CHAT_HISTORY_FILE, 'w') as f: json.dump(chats, f, indent=4)

# --- THE DATA SCIENTIST ENGINE ---
class DataAnalyst:
    @staticmethod
    def query_excel(filepath, question):
        """Uses Pandas to find exact answers in Excel/CSV"""
        try:
            # 1. Load Data with Smart Header Detection
            if filepath.endswith('.csv'): df = pd.read_csv(filepath, header=None, nrows=20)
            else: df = pd.read_excel(filepath, header=None, nrows=20)
            
            # Find header row (row with most columns)
            header_idx = 0
            max_cols = 0
            for i, row in df.iterrows():
                if row.count() > max_cols: max_cols = row.count(); header_idx = i
            
            # Reload with correct header
            if filepath.endswith('.csv'): df = pd.read_csv(filepath, header=header_idx)
            else: df = pd.read_excel(filepath, header=header_idx)

            # 2. KEYWORD SEARCH
            # Convert entire dataframe to string for searching
            df_str = df.astype(str).apply(lambda x: x.str.lower())
            q_terms = [w.lower() for w in question.split() if len(w) > 3]
            
            matched_rows = []
            for idx, row in df_str.iterrows():
                score = 0
                row_text = " ".join(row.values)
                for term in q_terms:
                    if term in row_text: score += 1
                if score > 0:
                    # Get original row data
                    orig_row = df.iloc[idx]
                    # Format as: "Column: Value | Column: Value"
                    row_str = " | ".join([f"{col}: {val}" for col, val in orig_row.items() if str(val).lower() != 'nan'])
                    matched_rows.append((score, row_str))
            
            # Sort by best match and take top 3
            matched_rows.sort(key=lambda x: x[0], reverse=True)
            results = [m[1] for m in matched_rows[:3]]
            
            return "\n".join(results)
        except: return ""

# --- READER (TEXT/PDF) ---
def read_text_file(filepath):
    text = ""
    try:
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".pdf":
            reader = PdfReader(filepath)
            text = "\n".join([p.extract_text() for p in reader.pages[:20]])
        elif ext == ".docx" and docx:
            doc = docx.Document(filepath)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif ext in [".txt", ".md", ".json"]:
            with open(filepath, "r", errors="ignore") as f: text = f.read()
    except: pass
    return text

# --- LOCAL BRAIN ---
class LocalBrain:
    def __init__(self):
        self.db = None
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.llm = ChatOllama(model=LOCAL_MODEL)
            if os.path.exists(os.path.join(DB_DIR, "index.faiss")):
                self.db = FAISS.load_local(DB_DIR, self.embeddings, allow_dangerous_deserialization=True)
        except: pass

    def is_alive(self):
        try: requests.get("http://localhost:11434"); return True
        except: return False

    def ingest(self, text, filepath):
        if not self.embeddings: return
        try:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_text(text)
            docs = [Document(page_content=c, metadata={"source": os.path.basename(filepath)}) for c in chunks]
            
            if self.db: self.db.add_documents(docs)
            else: self.db = FAISS.from_documents(docs, self.embeddings)
            self.db.save_local(DB_DIR)
        except: pass

    def ask(self, question, vault_path, chat_history):
        if not self.is_alive(): return "⚠️ Ollama is off. Open Ollama app."
        
        # 1. DATA ANALYST SCAN (Excel/CSV)
        data_context = ""
        for root, dirs, files in os.walk(vault_path):
            for file in files:
                if file.lower().endswith(('.xlsx', '.csv', '.xls')):
                    path = os.path.join(root, file)
                    results = DataAnalyst.query_excel(path, question)
                    if results:
                        data_context += f"\n[MATCH IN {file}]\n{results}\n"

        # 2. VECTOR SCAN (PDFs/Text)
        vector_context = ""
        if self.db:
            docs = self.db.similarity_search(question, k=4)
            vector_context = "\n".join([f"[SOURCE: {d.metadata['source']}]\n{d.page_content}" for d in docs])

        history_text = "\n".join([f"{msg['role']}: {msg['text']}" for msg in chat_history])
        
        prompt = f"""
        Answer the user's question using the DATA provided.
        
        --- EXCEL MATCHES (High Confidence) ---
        {data_context}
        
        --- TEXT/PDF MATCHES ---
        {vector_context}
        
        --- CHAT HISTORY ---
        {history_text}
        
        USER: {question}
        ANSWER:
        """
        
        try: return self.llm.invoke(prompt).content
        except Exception as e: return f"Error: {e}"

# --- HANDLER ---
class SimpleHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory: threading.Thread(target=self.process, args=(event.src_path,)).start()

    def process(self, filepath):
        config = load_config()
        filename = os.path.basename(filepath)
        if filename.startswith(".") or "crdownload" in filename: return
        time.sleep(2)
        
        # Sort logic
        text = read_text_file(filepath).lower() 
        target = "Others"
        for key, folder in config["semantic_rules"].items():
            if key in text or key in filename.lower(): target = folder; break
        
        if target == "Others":
            ext = os.path.splitext(filename)[1].lower()
            for folder, exts in config["extension_rules"].items():
                if ext in exts: target = folder; break

        # Rename (AI)
        new_name = filename
        if config.get("ai_renaming") and len(text) > 10:
            try:
                llm = ChatOllama(model=LOCAL_MODEL)
                resp = llm.invoke(f"Rename file concisely (Type_Entity_Date) based on: {text[:300]}")
                clean = resp.content.strip().replace(" ", "_").replace("/", "-")
                if len(clean) < 50: 
                    ext = os.path.splitext(filename)[1]
                    new_name = clean + ext
            except: pass

        dest = os.path.join(config["target_dir"], target)
        os.makedirs(dest, exist_ok=True)
        final = os.path.join(dest, new_name)
        
        count = 1
        base, ext = os.path.splitext(filename)
        while os.path.exists(final):
            final = os.path.join(dest, f"{base}_{count}{ext}")
            count += 1
            
        try:
            shutil.move(filepath, final)
            if config.get("rag_enabled") and text:
                LocalBrain().ingest(text, final)
        except: pass

# --- GUI ---
class DashboardWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME} {VERSION}")
        self.geometry("1000x650")
        self.config = load_config()
        self.current_chat_id = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        ctk.CTkLabel(self.sidebar, text="Chat History", font=("Arial", 18, "bold")).pack(pady=20)
        ctk.CTkButton(self.sidebar, text="+ New Chat", command=self.new_chat, fg_color="#2cc985", text_color="black").pack(pady=10, padx=10)
        self.scroll_history = ctk.CTkScrollableFrame(self.sidebar, fg_color="transparent")
        self.scroll_history.pack(fill="both", expand=True)
        
        self.main_area = ctk.CTkTabview(self)
        self.main_area.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.tab_chat = self.main_area.add("🧠 Chat with Nova")
        self.tab_settings = self.main_area.add("⚙️ Settings")

        self.build_chat_ui()
        self.build_settings_ui()
        self.load_sidebar()
        
        self.brain = LocalBrain()

    def build_chat_ui(self):
        # 1. Chat Display (Top)
        self.chat_display = ctk.CTkTextbox(self.tab_chat, state="disabled", font=("Arial", 14), wrap="word")
        self.chat_display.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 2. Input Area (Bottom - Pinned)
        input_frame = ctk.CTkFrame(self.tab_chat, height=50)
        input_frame.pack(fill="x", padx=10, pady=10, side="bottom")
        
        self.chat_input = ctk.CTkEntry(input_frame, placeholder_text="Ask about your files...")
        self.chat_input.pack(side="left", fill="x", expand=True, padx=5, pady=5)
        self.chat_input.bind("<Return>", lambda e: self.send_msg())
        
        ctk.CTkButton(input_frame, text="Send", command=self.send_msg, width=60).pack(side="right", padx=5, pady=5)

    def load_sidebar(self):
        for widget in self.scroll_history.winfo_children(): widget.destroy()
        chats = ChatManager.load_chats()
        for chat_id in reversed(list(chats.keys())):
            btn = ctk.CTkButton(self.scroll_history, text=chat_id, command=lambda c=chat_id: self.load_chat_session(c), fg_color="transparent", border_width=1, text_color=("gray10", "gray90"))
            btn.pack(pady=2, padx=5, fill="x")

    def new_chat(self):
        self.current_chat_id = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.chat_display.configure(state="normal")
        self.chat_display.delete("0.0", "end")
        self.chat_display.configure(state="disabled")
        self.load_sidebar()

    def load_chat_session(self, chat_id):
        self.current_chat_id = chat_id
        chats = ChatManager.load_chats()
        history = chats.get(chat_id, [])
        self.chat_display.configure(state="normal")
        self.chat_display.delete("0.0", "end")
        for msg in history:
            prefix = "You" if msg['role'] == 'User' else "AI"
            self.chat_display.insert("end", f"{prefix}: {msg['text']}\n\n")
        self.chat_display.configure(state="disabled")
        self.chat_display.see("end")

    def send_msg(self):
        msg = self.chat_input.get()
        if not msg: return
        self.chat_input.delete(0, "end")
        
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", f"You: {msg}\n")
        self.chat_display.insert("end", "Thinking...\n\n")
        self.chat_display.configure(state="disabled")
        self.chat_display.see("end")
        
        def run():
            chats = ChatManager.load_chats()
            history = chats.get(self.current_chat_id, [])
            ans = self.brain.ask(msg, self.config["target_dir"], history)
            history.append({"role": "User", "text": msg})
            history.append({"role": "Nova AI", "text": ans})
            ChatManager.save_chat(self.current_chat_id, history)
            
            self.chat_display.configure(state="normal")
            self.chat_display.delete("end-2l", "end") 
            self.chat_display.insert("end", f"AI: {ans}\n\n{'-'*30}\n\n")
            self.chat_display.configure(state="disabled")
            self.after(0, self.load_sidebar)

        threading.Thread(target=run).start()

    def build_settings_ui(self):
        ctk.CTkLabel(self.tab_settings, text="Local AI Settings (Ollama)").pack(pady=10)
        self.var_ai = ctk.BooleanVar(value=self.config.get("ai_renaming", True))
        ctk.CTkSwitch(self.tab_settings, text="Enable Local AI Rename", variable=self.var_ai).pack(pady=10)
        ctk.CTkButton(self.tab_settings, text="Save Config", command=self.save, fg_color="green").pack(pady=20)
        ctk.CTkButton(self.tab_settings, text="🔄 Re-Build Local Index", command=self.rescan, fg_color="#F39C12").pack(pady=10)

    def rescan(self):
        threading.Thread(target=self._run_rescan).start()
        messagebox.showinfo("Started", "Building Brain... (Runs offline)")

    def _run_rescan(self):
        brain = LocalBrain()
        vault = self.config["target_dir"]
        if os.path.exists(DB_DIR): shutil.rmtree(DB_DIR)
        os.makedirs(DB_DIR, exist_ok=True)
        for root, dirs, files in os.walk(vault):
            for file in files:
                if file.lower().endswith(('.pdf', '.txt', '.docx')):
                    path = os.path.join(root, file)
                    text = read_text_file(path)
                    if text: brain.ingest(text, path)
        logging.info("Index Complete")

    def save(self):
        self.config["ai_renaming"] = self.var_ai.get()
        save_config(self.config)
        messagebox.showinfo("Saved", "Config Saved!")

if __name__ == "__main__":
    if "--settings" in sys.argv:
        app = DashboardWindow()
        app.mainloop()
    else:
        app = DashboardWindow()
        app.mainloop()
