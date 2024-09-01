# Corrective-RAG (CRAG) API
### Overview

The Corrective-RAG (CRAG) API is designed to automatically generate educational materials such as quizzes, tests, and exams using AI. It leverages LangChain, DRF, and local LLMs to build an intelligent retrieval-augmented generation (RAG) system that incorporates corrective mechanisms to ensure the relevance and quality of the retrieved documents.

### Key Features
- **Document Indexing & Retrieval**: Indexes documents and retrieves relevant information based on user queries.
- **Corrective RAG Strategy**: Incorporates self-reflection/self-grading on retrieved documents. If irrelevant documents are identified, the system supplements the retrieval process with web searches using Tavily Search.
- **AI-Generation**: Generates educational materials.

### General Flow
1. Document Indexing: The API indexes documents provided by the user.
2. RAG Process: The API performs retrieval-augmented generation (RAG) on indexed documents based on the user's query.
3. Correction Mechanism: If irrelevant documents are retrieved, a web search is performed to supplement the retrieval process.
4. Content Generation: The system generates content based on the refined information.
5. Tracing: For tracing and testing I used LangSmith.

### API Endpoint
Endpoint: POST /llm/api/answer/

**Request Body**:

```json
{
    "question": "I want you to create 2 Multiple Choice Questions that test the knowledge of LLM-powered autonomous agent system's key components."
}
```

**Response**:

```json
{
    "response": "1. Question: Which key component of an LLM-powered autonomous agent system functions as the agent's brain?\n       - A) The expert models\n       - B) The API tools\n       - C) The LLM\n       - D) The prompting framework\n\n   Answer Explanation: The LLM is the key component that functions as the agent's brain in an LLM-powered autonomous agent system. This is based on the information from the documents provided, where it is stated that \"In a LLM-powered autonomous agent system, LLM functions as the agent's brain.\"\n\n  2. Question: What does the LLM do when presented with a complex task in an LLM-powered autonomous agent system?\n       - A) It fetches the relevant data using APIs\n       - B) It distributes the tasks to expert models\n       - C) It generates well-written content\n       - D) It presents a history of sequentially improved outputs in context and trains the model to take on the trend to produce better outputs\n\n   Answer Explanation: When presented with a complex task, the LLM distributes the tasks to expert models. This is based on the information from the documents provided, where it is stated that \"Due to the limited context length, task type based filtration is used to distribute the tasks to expert models.\"",
    "steps": [
        "retrieve_documents",
        "grade_document_retrieval",
        "web_search",
        "generate_answer"
    ]
}
```

**Explanation**:

- **response**: Contains the generated multiple-choice questions and answer explanations.
- **steps**: Outlines the steps taken by the API to produce the response, including document retrieval, grading, web search (if necessary), and answer generation.

**Installation & Setup**

**Prerequisites**

- **Python 3.8+**
- **Ollama App**: Download and install the Ollama app.
- **LangChain & DRF**: The API is built using LangChain and Django Rest Framework (DRF).
- **Virtual Environment**: Install dependencies within a virtual environment.

**Installation Steps**

1. **Clone the Repository**:

```bash
git clone <repository-url>
cd <repository-directory>
```
2. **Set Up the Virtual Environment**:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install Dependencies**:

```bash
pip install -r requirements.txt
```

4. **Download and Configure the Model**:

- Pull the required model using Ollama:
```bash
ollama pull mistral:instruct
```

5. **Set Up Environment Variables**:

- Create a .env file and add the following variables:
```env
TAVILY_API_KEY=<your-tavily-api-key>
LANGCHAIN_API_KEY=<your-langchain-api-key>
LANGCHAIN_PROJECT=<your-langchain-project-name>
LOCAL_LLM=mistral:instruct
```

6. **Run the Application**:

```bash
python manage.py runserver
```

