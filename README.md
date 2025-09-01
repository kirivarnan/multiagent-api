# multiagent-api

# Multi-Agent AI System FastAPI

A comprehensive REST API for running multi-agent workflows with researcher, analyst, and writer agents powered by Groq LLM.

## 🏗️ Project Structure

```
project/
├── main.py                           # FastAPI main application
├── requirements.txt                  # Dependencies
├── .env.example                     # Environment variables template
├── README.md                        # Documentation
├── controllers/
│   ├── __init__.py
│   └── multiagent_controller.py     # API endpoints and validation
├── services/
│   ├── __init__.py
│   ├── multiagent_service.py        # Business logic layer
│   └── multiagent_workflow.py       # Original workflow code
```

## 🚀 Features

### Multi-Agent Workflow

- **Supervisor**: Orchestrates the workflow and decides next agent
- **Researcher**: Gathers comprehensive information on topics
- **Analyst**: Analyzes data and provides strategic insights
- **Writer**: Creates professional reports and summaries

### API Features

- **Synchronous Processing**: Immediate response with results
- **Asynchronous Processing**: Background processing for long tasks
- **Task Status Tracking**: Monitor progress of background tasks
- **Automatic Documentation**: Interactive API docs
- **Data Validation**: Request/response validation with Pydantic
- **Error Handling**: Comprehensive error responses
- **CORS Support**: Frontend integration ready

## 📋 API Endpoints

### Core Endpoints

- **POST** `/api/v1/process` - Process query through workflow
- **POST** `/api/v1/process-async` - Start async processing
- **GET** `/api/v1/tasks/{task_id}/status` - Check task status

### System Endpoints

- **GET** `/api/v1/agents/status` - Get agent status
- **GET** `/api/v1/health` - Health check
- **GET** `/` - API information

### Legacy

- **POST** `/api/v1/ask` - Legacy endpoint (deprecated)

## 🛠️ Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Add your Groq API key to .env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Run the Application

```bash
# Development mode
python main.py

# Or with uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Access Documentation

- **Interactive API Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## 🔧 Usage Examples

### Synchronous Processing

```bash
curl -X POST "http://localhost:8000/api/v1/process" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the benefits and risks of AI in healthcare?",
    "include_full_response": true
  }'
```

### Asynchronous Processing

```bash
# Start processing
curl -X POST "http://localhost:8000/api/v1/process-async" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Analyze the future of renewable energy"
  }'

# Check status (use task_id from response)
curl "http://localhost:8000/api/v1/tasks/{task_id}/status"
```

### Python Client Example

```python
import requests

# Process a query
response = requests.post(
    "http://localhost:8000/api/v1/process",
    json={
        "query": "What are the latest trends in artificial intelligence?",
        "include_full_response": False
    }
)

result = response.json()
print(result["final_report"])
```

## 📊 Response Format

### Synchronous Response

```json
{
  "task_id": "uuid-string",
  "query": "Your question",
  "status": "completed",
  "final_report": "Generated report...",
  "messages": [...],
  "research_data": "...",
  "analysis": "...",
  "processing_time": 15.2,
  "created_at": "2024-01-01T12:00:00",
  "completed_at": "2024-01



docker login
docker tag multiagent-api githubkiri/mutliagent-api:v1
docker build -t githubkiri/multiagent-api .
docker push githubkiri/multiagent-api
docker run -d -p 8060:8060 --name multiagent-api githubkiri/multiagent-api

curl -X POST "https://multiagent-api-latest.onrender.com/api/v1/ask" -H "Content-Type: application/json" -d "{\"query\": \"what is agentic-ai and pros\", \"include_full_response\": false}"


```
