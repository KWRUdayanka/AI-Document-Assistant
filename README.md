# 🧠 AI Document Assistant

This script handles the ingestion of `.pdf`, `.docx`, and `.txt` documents into a Qdrant vector database using LlamaIndex, OpenAI embeddings, and QdrantClient.

A **Retrieval-Augmented Generation (RAG)** chatbot built using:

- 🧪 FastAPI for serving queries
- 📦 Qdrant for semantic vector storage
- 🤖 OpenAI for embeddings and GPT-powered answers
- 🧠 LlamaIndex for context-based answer synthesis
- 🎨 Custom logging for clear CLI feedback

---

![screenshot](docs/Screenshot 2025-08-01 230800.png)
![screenshot](docs/Screenshot 2025-08-01 230854.png)

---

## 📂 Folder Structure

```
project-root/
│
├── data/                     # Place your documents here (PDF, DOCX, TXT)
├── src/
│   └── ingest.py             # Main ingestion script
│   └── query.py              # Chat interface / query engine
│   └── utils.py              # Helper functions
│   └── custom_logging.py     # Logging setup (if used)
├── requirements.txt
├── .env.example
├── .env                      # Environment variables
├── README.md
└── run_chatbot.py            # Entry point for demo
```
## 🧬 Environment Setup
- Clone this repository
```bash
Copy
Edit
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

---
## ⚙️ Environment Variables (`.env`)

```env
QDRANT_COLLECTION=dfn_collection_name
OPENAI_API_KEY=your_openai_key
```

---

## 🚀 Setup & Run

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Prepare your `.env` and `data/` folder.**

3. **Run the ingestion script**

```bash
python src/ingest.py
```

4. **Run the run_chatbot script**

```bash
python run_chatbot.py
```

---

## 🧰 **Technologies Used**

| Component  | Description                     |
|------------|---------------------------------|
| FastAPI    | REST API framework              |
| Qdrant    | Vector similarity search engine  |
| OpenAI    | Embedding and LLM provider       |
| LlamaIndex | RAG engine and response synthesizer |
| Uvicorn   | ASGI server                     |
| Dotenv    | Environment config              |
| Colorlog  | Colorful terminal logging       |

---

## 💬 Query the API

- You can now send POST requests to the API to ask questions.
## Example using curl:
```
curl --location 'http://127.0.0.1:8000/query' \
--header 'Content-Type: application/json' \
--data '{
    "text": "How do I use Qdrant?"
}'
```
## Or use Postman:
- Method: POST
- URL: http://127.0.0.1:8000/query
- Body: raw → JSON
```
{
  "text": "How do I use Qdrant?"
}
```

---

## 🚀 Features

- ✅ Supports `.pdf`, `.docx`, and `.txt` files
- ✅ Chunks documents into manageable token sizes
- ✅ Embeds each chunk using OpenAI
- ✅ Stores chunks with metadata in Qdrant
- ✅ Auto-creates collection if it doesn’t exist
- 🔍 Semantic document search (OpenAI embeddings + Qdrant)
- 🧠 Contextual LLM-based answering with LlamaIndex
- 🚀 FastAPI REST API endpoint: `POST /query`
- ✅ Colorful logs for easy debugging and clarity

---

## 📦 Qdrant Setup (Optional)

Ensure Qdrant is running on `localhost`. If you don’t have it:

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

---

## .env

- QDRANT_COLLECTION=your_collection_name
- OPENAI_API_KEY=your_openai_api_key
- JWT_SECRET_KEY=your_secret_key


---

## 🧪 Output

- Logs progress, errors, and chunk details
- Embedding vectors and metadata are stored in Qdrant collection defined in `.env`

---

## 🛠️ Tech Stack

- Python 3.10+
- FastAPI
- LlamaIndex
- Qdrant
- OpenAI (Embeddings + GPT)
- PyMuPDF, python-docx
- colorlog

---

## 📥 Output & Logs

- Progress and errors are printed with color-coded logs.
- Embedded documents are stored in your specified Qdrant collection.
---

## 📄 Requirements

You can generate the dependencies list with:

```bash
pip freeze > requirements.txt
```

Sample minimum requirements:

```txt
openai
llama-index
qdrant-client
python-dotenv
colorlog
PyMuPDF
python-docx
tqdm
fastapi
uvicorn
```
---

## 📦 Installation

```bash
git clone https://github.com/yourname/ai-document-assistant.git
cd ai-document-assistant
pip install -r requirements.txt
cp .env.example .env  # Add your OpenAI key here
```

---

## 📌 License

- MIT or customize as needed.

---

## 🗃️ Query Flow

- User sends a query via /query.
- Embedding generated using OpenAIEmbedding.
- Vector search in Qdrant.
- Top K documents retrieved.
- Documents passed to LlamaIndex synthesizer.
- Response returned to user.

---

## 🙋‍♂️ Author

- Created by [Rashan Udayanka] — AI + RAG developer.

---

## 📌 To-Do
- Add authentication with JWT
- Dockerize the app
- Add unit tests
- Frontend for chat