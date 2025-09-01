from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any, Literal
from datetime import datetime
import uuid
import asyncio
import os
from dotenv import load_dotenv

# Import your existing multiagent dependencies
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq

# ===================================
# FastAPI App Setup
# ===================================
app = FastAPI(
    title="Multi-Agent AI System API",
    description="A REST API for running multi-agent workflows with researcher, analyst, and writer agents",
    version="1.0.0"
)

# ===================================
# Pydantic Models
# ===================================
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

# ===================================
# Multi-Agent Workflow (Your Original Code)
# ===================================
class SupervisorState(MessagesState):
    """State for the multi-agent system"""
    next_agent: str = ""
    research_data: str = ""
    analysis: str = ""
    final_report: str = ""
    task_complete: bool = False
    current_task: str = ""

load_dotenv()
memory = MemorySaver()
llm = ChatGroq(model="llama-3.1-8b-instant")

def create_supervisor_chain():
    """Creates the supervisor decision chain"""
    supervisor_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a supervisor managing a team of agents:
        
1. Researcher - Gathers information and data
2. Analyst - Analyzes data and provides insights  
3. Writer - Creates reports and summaries

Based on the current state and conversation, decide which agent should work next.
If the task is complete, respond with 'DONE'.

Current state:
- Has research data: {has_research}
- Has analysis: {has_analysis}
- Has report: {has_report}

Respond with ONLY the agent name (researcher/analyst/writer) or 'DONE'.
"""),
        ("human", "{task}")
    ])
    return supervisor_prompt | llm

def supervisor_agent(state: SupervisorState) -> Dict:
    """Supervisor decides next agent using Groq LLM"""
    messages = state["messages"]
    task = messages[-1].content if messages else "No task"
    
    has_research = bool(state.get("research_data", ""))
    has_analysis = bool(state.get("analysis", ""))
    has_report = bool(state.get("final_report", ""))
    
    chain = create_supervisor_chain()
    decision = chain.invoke({
        "task": task,
        "has_research": has_research,
        "has_analysis": has_analysis,
        "has_report": has_report
    })
    
    decision_text = decision.content.strip().lower()
    
    if "done" in decision_text or has_report:
        next_agent = "end"
        supervisor_msg = "✅ Supervisor: All tasks complete! Great work team."
    elif "researcher" in decision_text or not has_research:
        next_agent = "researcher"
        supervisor_msg = "📋 Supervisor: Let's start with research. Assigning to Researcher..."
    elif "analyst" in decision_text or (has_research and not has_analysis):
        next_agent = "analyst"
        supervisor_msg = "📋 Supervisor: Research done. Time for analysis. Assigning to Analyst..."
    elif "writer" in decision_text or (has_analysis and not has_report):
        next_agent = "writer"
        supervisor_msg = "📋 Supervisor: Analysis complete. Let's create the report. Assigning to Writer..."
    else:
        next_agent = "end"
        supervisor_msg = "✅ Supervisor: Task seems complete."
    
    return {
        "messages": [AIMessage(content=supervisor_msg)],
        "next_agent": next_agent,
        "current_task": task
    }

def researcher_agent(state: SupervisorState) -> Dict:
    """Researcher uses Groq to gather information"""
    task = state.get("current_task", "research topic")
    
    research_prompt = f"""As a research specialist, provide comprehensive information about: {task}

    Include:
    1. Key facts and background
    2. Current trends or developments
    3. Important statistics or data points
    4. Notable examples or case studies
    
    Be concise but thorough."""
    
    research_response = llm.invoke([HumanMessage(content=research_prompt)])
    research_data = research_response.content
    
    agent_message = f"🔍 Researcher: I've completed the research on '{task}'.\n\nKey findings:\n{research_data[:500]}..."
    
    return {
        "messages": [AIMessage(content=agent_message)],
        "research_data": research_data,
        "next_agent": "supervisor"
    }

def analyst_agent(state: SupervisorState) -> Dict:
    """Analyst uses Groq to analyze the research"""
    research_data = state.get("research_data", "")
    task = state.get("current_task", "")
    
    analysis_prompt = f"""As a data analyst, analyze this research data and provide insights:

Research Data:
{research_data}

Provide:
1. Key insights and patterns
2. Strategic implications
3. Risks and opportunities
4. Recommendations

Focus on actionable insights related to: {task}"""
    
    analysis_response = llm.invoke([HumanMessage(content=analysis_prompt)])
    analysis = analysis_response.content
    
    agent_message = f"📊 Analyst: I've completed the analysis.\n\nTop insights:\n{analysis[:400]}..."
    
    return {
        "messages": [AIMessage(content=agent_message)],
        "analysis": analysis,
        "next_agent": "supervisor"
    }

def writer_agent(state: SupervisorState) -> Dict:
    """Writer uses Groq to create final report"""
    research_data = state.get("research_data", "")
    analysis = state.get("analysis", "")
    task = state.get("current_task", "")
    
    writing_prompt = f"""As a professional writer, create an executive report based on:

Task: {task}

Research Findings:
{research_data[:1000]}

Analysis:
{analysis[:1000]}

Create a well-structured report with:
1. Executive Summary
2. Key Findings  
3. Analysis & Insights
4. Recommendations
5. Conclusion

Keep it professional and concise."""
    
    report_response = llm.invoke([HumanMessage(content=writing_prompt)])
    report = report_response.content
    
    final_report = f"""
📄 FINAL REPORT
{'='*50}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Topic: {task}
{'='*50}

{report}

{'='*50}
Report compiled by Multi-Agent AI System powered by Groq
"""
    
    return {
        "messages": [AIMessage(content=f"✍️ Writer: Report complete!")],
        "final_report": final_report,
        "next_agent": "supervisor",
        "task_complete": True
    }

def router(state: SupervisorState) -> Literal["supervisor", "researcher", "analyst", "writer", "__end__"]:
    """Routes to next agent based on state"""
    next_agent = state.get("next_agent", "supervisor")
    
    if next_agent == "end" or state.get("task_complete", False):
        return END
        
    if next_agent in ["supervisor", "researcher", "analyst", "writer"]:
        return next_agent
        
    return "supervisor"

# Create workflow
workflow = StateGraph(SupervisorState)
workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("analyst", analyst_agent)
workflow.add_node("writer", writer_agent)
workflow.set_entry_point("supervisor")

for node in ["supervisor", "researcher", "analyst", "writer"]:
    workflow.add_conditional_edges(
        node,
        router,
        {
            "supervisor": "supervisor",
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            END: END
        }
    )

graph = workflow.compile()

# ===================================
# Service Class
# ===================================
class MultiAgentService:
    def __init__(self):
        self.active_tasks = {}
    
    async def process_query(self, query: str, task_id: str) -> Dict[str, Any]:
        """Process a query through the multi-agent workflow"""
        try:
            # Run workflow in executor to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: graph.invoke({"messages": [HumanMessage(content=query)]})
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Workflow error: {str(e)}")
    
    async def process_query_background(self, query: str, task_id: str):
        """Process query in background"""
        try:
            self.active_tasks[task_id] = {
                "status": "processing",
                "query": query,
                "created_at": datetime.now(),
                "progress": "Starting workflow..."
            }
            
            result = await self.process_query(query, task_id)
            
            self.active_tasks[task_id].update({
                "status": "completed",
                "result": result,
                "completed_at": datetime.now()
            })
        except Exception as e:
            self.active_tasks[task_id].update({
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.now()
            })
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status"""
        return self.active_tasks.get(task_id)

# Initialize service
multiagent_service = MultiAgentService()

# ===================================
# API Endpoints
# ===================================
@app.post("/api/v1/process", response_model=WorkflowResponse)
async def process_query(request: QueryRequest):
    """Process a query through the multi-agent workflow"""
    task_id = str(uuid.uuid4())
    
    try:
        start_time = datetime.now()
        result = await multiagent_service.process_query(request.query, task_id)
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
        
        return WorkflowResponse(
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
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )

@app.post("/api/v1/process-async", status_code=status.HTTP_202_ACCEPTED)
async def process_query_async(request: QueryRequest, background_tasks: BackgroundTasks):
    """Start processing a query asynchronously"""
    task_id = str(uuid.uuid4())
    
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

@app.get("/api/v1/tasks/{task_id}/status")
async def get_task_status(task_id: str):
    """Get the status of a background task"""
    status_info = await multiagent_service.get_task_status(task_id)
    if not status_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    return status_info

@app.get("/api/v1/agents/status")
async def get_agents_status():
    """Get status of all available agents"""
    return [
        {"agent_name": "supervisor", "status": "active", "description": "Orchestrates workflow"},
        {"agent_name": "researcher", "status": "active", "description": "Gathers information"},
        {"agent_name": "analyst", "status": "active", "description": "Analyzes data"},
        {"agent_name": "writer", "status": "active", "description": "Creates reports"}
    ]

@app.get("/api/v1/health")
async def health_check():
    """Health check for multi-agent system"""
    return {
        "status": "healthy",
        "service": "Multi-Agent AI System",
        "agents": ["supervisor", "researcher", "analyst", "writer"],
        "uptime": "active"
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Multi-Agent AI System API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "process_query": "/api/v1/process",
            "process_async": "/api/v1/process-async",
            "health": "/api/v1/health",
            "agents_status": "/api/v1/agents/status"
        }
    }

# Legacy endpoint for backward compatibility
@app.post("/api/v1/ask")
async def ask_agent(request: QueryRequest):
    """Legacy endpoint - use /process instead"""
    return await process_query(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)