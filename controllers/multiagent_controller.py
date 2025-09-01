from fastapi import APIRouter, HTTPException, BackgroundTasks, status
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime
import uuid
import asyncio
from services.multiagent_service import MultiAgentService

router = APIRouter(tags=["multi-agent"])

# Pydantic models for request/response validation
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="The query to process")
    include_full_response: bool = Field(default=False, description="Include full workflow response")

class AgentMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime

class WorkflowResponse(BaseModel):
    task_id: str
    query: str
    status: str
    final_report: Optional[str] = None
    messages: List[AgentMessage] = []
    research_data: Optional[str] = None
    analysis: Optional[str] = None
    processing_time: Optional[float] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

class ErrorResponse(BaseModel):
    error: str
    task_id: Optional[str] = None
    timestamp: datetime

class HealthResponse(BaseModel):
    status: str
    service: str
    agents: List[str]
    uptime: str

class AgentStatusResponse(BaseModel):
    agent_name: str
    status: str
    description: str

# Initialize service
multiagent_service = MultiAgentService()

@router.post("/process", response_model=WorkflowResponse, status_code=status.HTTP_200_OK)
async def process_query(request: QueryRequest):
    """
    Process a query through the multi-agent workflow
    
    - **query**: The question or task to process
    - **include_full_response**: Whether to include detailed workflow data
    """
    task_id = str(uuid.uuid4())
    
    try:
        start_time = datetime.now()
        
        # Process the query
        result = await multiagent_service.process_query(
            query=request.query,
            task_id=task_id
        )
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Format messages
        messages = []
        if "messages" in result:
            for msg in result["messages"]:
                messages.append(AgentMessage(
                    role=getattr(msg, 'type', 'system'),
                    content=str(msg.content),
                    timestamp=datetime.now()
                ))
        
        response = WorkflowResponse(
            task_id=task_id,
            query=request.query,
            status="completed",
            final_report=result.get("final_report"),
            messages=messages if request.include_full_response else [],
            research_data=result.get("research_data") if request.include_full_response else None,
            analysis=result.get("analysis") if request.include_full_response else None,
            processing_time=processing_time,
            created_at=start_time,
            completed_at=end_time
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )

@router.post("/process-async", status_code=status.HTTP_202_ACCEPTED)
async def process_query_async(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    Start processing a query asynchronously (for long-running tasks)
    
    Returns a task ID that can be used to check status
    """
    task_id = str(uuid.uuid4())
    
    # Add to background tasks
    background_tasks.add_task(
        multiagent_service.process_query_background,
        request.query,
        task_id
    )
    
    return {
        "task_id": task_id,
        "status": "processing",
        "message": "Query processing started",
        "check_status_url": f"/api/v1/tasks/{task_id}/status"
    }

@router.get("/tasks/{task_id}/status")
async def get_task_status(task_id: str):
    """Get the status of a background task"""
    try:
        status_info = await multiagent_service.get_task_status(task_id)
        if not status_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found"
            )
        return status_info
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting task status: {str(e)}"
        )

@router.get("/agents/status", response_model=List[AgentStatusResponse])
async def get_agents_status():
    """Get status of all available agents"""
    agents_status = [
        AgentStatusResponse(
            agent_name="supervisor",
            status="active",
            description="Orchestrates workflow and decides next agent"
        ),
        AgentStatusResponse(
            agent_name="researcher",
            status="active",
            description="Gathers information and data on topics"
        ),
        AgentStatusResponse(
            agent_name="analyst",
            status="active",
            description="Analyzes data and provides insights"
        ),
        AgentStatusResponse(
            agent_name="writer",
            status="active",
            description="Creates reports and summaries"
        )
    ]
    return agents_status

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check for multi-agent system"""
    return HealthResponse(
        status="healthy",
        service="Multi-Agent AI System",
        agents=["supervisor", "researcher", "analyst", "writer"],
        uptime="active"
    )

# Legacy endpoint for backward compatibility
@router.post("/ask")
async def ask_agent(request: QueryRequest):
    """
    Legacy endpoint - redirects to /process
    [DEPRECATED] Use /process instead
    """
    return await process_query(request)