from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from controllers.multiagent_controller import router as multiagent_router

app = FastAPI(
    title="Multi-Agent AI System API",
    description="A REST API for running multi-agent workflows with researcher, analyst, and writer agents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(multiagent_router, prefix="/api/v1")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Multi-Agent AI System API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "process_query": "/api/v1/process",
            "health": "/api/v1/health",
            "agents_status": "/api/v1/agents/status"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Multi-Agent AI System",
        "agents": ["supervisor", "researcher", "analyst", "writer"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


"""
docker build -t multiagent-api .
docker run -d -p 8060:8060 --name multiagent-api multiagent-api
curl http://localhost:8060/
curl http://localhost:8060/api/v1/health
curl -X POST "http://localhost:8060/api/v1/ask" -H "Content-Type: application/json" -d "{\"query\": \"what is agentic ai and pros\", \"include_full_response\": false}"
"""