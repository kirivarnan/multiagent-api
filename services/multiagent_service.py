import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
from langchain_core.messages import HumanMessage
import logging

# Import your existing multiagent code
from .multiagent_workflow import run_multiagent_workflow

logger = logging.getLogger(__name__)

class MultiAgentService:
    """Service class for handling multi-agent workflow operations"""
    
    def __init__(self):
        self.active_tasks = {}  # Store background task status
    
    async def process_query(self, query: str, task_id: str) -> Dict[str, Any]:
        """
        Process a query through the multi-agent workflow
        
        Args:
            query: The user query to process
            task_id: Unique identifier for this task
            
        Returns:
            Dict containing the workflow results
        """
        try:
            logger.info(f"Processing query for task {task_id}: {query[:100]}...")
            
            # Run the multiagent workflow
            # Note: If your workflow is not async, we wrap it
            result = await asyncio.get_event_loop().run_in_executor(
                None, run_multiagent_workflow, query
            )
            
            logger.info(f"Successfully processed task {task_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing task {task_id}: {str(e)}")
            raise e
    
    async def process_query_background(self, query: str, task_id: str):
        """
        Process a query in the background for async endpoints
        
        Args:
            query: The user query to process
            task_id: Unique identifier for this task
        """
        try:
            # Update task status to processing
            self.active_tasks[task_id] = {
                "status": "processing",
                "query": query,
                "created_at": datetime.now(),
                "progress": "Starting workflow..."
            }
            
            # Process the query
            result = await self.process_query(query, task_id)
            
            # Update task status to completed
            self.active_tasks[task_id].update({
                "status": "completed",
                "result": result,
                "completed_at": datetime.now(),
                "progress": "Workflow completed successfully"
            })
            
        except Exception as e:
            # Update task status to failed
            self.active_tasks[task_id].update({
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.now(),
                "progress": f"Workflow failed: {str(e)}"
            })
            logger.error(f"Background task {task_id} failed: {str(e)}")
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a background task
        
        Args:
            task_id: The task identifier
            
        Returns:
            Dict containing task status information
        """
        task_info = self.active_tasks.get(task_id)
        if not task_info:
            return None
            
        return {
            "task_id": task_id,
            "status": task_info["status"],
            "query": task_info["query"],
            "created_at": task_info["created_at"],
            "completed_at": task_info.get("completed_at"),
            "progress": task_info.get("progress"),
            "result": task_info.get("result"),
            "error": task_info.get("error")
        }
    
    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """
        Clean up old completed tasks to prevent memory buildup
        
        Args:
            max_age_hours: Maximum age in hours before cleanup
        """
        current_time = datetime.now()
        tasks_to_remove = []
        
        for task_id, task_info in self.active_tasks.items():
            task_age = current_time - task_info["created_at"]
            if task_age.total_seconds() > (max_age_hours * 3600):
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del self.active_tasks[task_id]
            logger.info(f"Cleaned up old task: {task_id}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics
        
        Returns:
            Dict containing system stats
        """
        total_tasks = len(self.active_tasks)
        completed_tasks = sum(1 for task in self.active_tasks.values() if task["status"] == "completed")
        processing_tasks = sum(1 for task in self.active_tasks.values() if task["status"] == "processing")
        failed_tasks = sum(1 for task in self.active_tasks.values() if task["status"] == "failed")
        
        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "processing_tasks": processing_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        }