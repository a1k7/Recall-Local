# Recall: The Corporate Brain 🧠📂

**The Privacy-First Local File Organizer & Intelligence Engine.**

SmartSort AI is a powerful desktop application that automatically organizes your files and allows you to "chat" with your documents using a local AI model. It runs 100% offline using **Ollama**, ensuring your sensitive data never leaves your device.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows-lightgrey) ![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

### 🗂️ Automated Organization
* **Watchdog Engine:** Silently monitors your `Downloads` and `Desktop` folders.
* **Smart Sorting:** Automatically moves files into a structured Vault (`Documents/SmartSort_Vault`) based on their content (e.g., *Invoices* $\rightarrow$ *Financial*).
* **AI Renaming:** Intelligently renames messy files like `scan_291.pdf` to `Invoice_Vendor_Date.pdf`.

### 🧠 Local Intelligence (RAG)
* **Chat with Data:** Ask questions like *"How much did I pay Ola?"* or *"Who is assigned the content task?"*.
* **Data Scientist Engine:** specialized logic to read **Excel & CSV** files row-by-row for accurate data extraction.
* **Context Memory:** Remembers your conversation context (e.g., understands who "she" refers to).
* **Privacy First:** Powered by **Llama 3.2** via Ollama. No API keys required. No cloud uploads.

### 🖥️ Modern UI
* **Dashboard:** Clean, dark-mode interface with chat history.
* **System Tray:** Runs quietly in the background with a menu bar icon.

---

## 🚀 Installation & Setup

### 1. Prerequisites
You need **[Ollama](https://ollama.com/)** installed and running on your computer to power the AI features.
```bash
# Install the model (run this in terminal)
ollama run llama3.2

<img width="2000" height="1600" alt="Code’s output demo" src="https://github.com/user-attachments/assets/71dcf24f-45f5-45e8-bc55-f8aff3101ecb" />
