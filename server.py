#!/usr/bin/env python3
"""
FastAPI + Socket.IO server for DSPy prompt optimization UI
Provides real-time updates during optimization and evaluation processes
"""

import asyncio
import json
import os
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import socketio
import uvicorn

# Create Socket.IO server
sio = socketio.AsyncServer(cors_allowed_origins="*", async_mode="asgi")

# Create FastAPI app
app = FastAPI(title="DSPy Optimization API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Combine FastAPI and Socket.IO
socket_app = socketio.ASGIApp(sio, app)

# Global optimization state
optimization_state = {
    "status": "idle",  # idle, running, completed, error
    "progress": 0,
    "current_step": "",
    "results": {},
    "session_id": None,
    "start_time": None,
    "end_time": None
}

@sio.event
async def connect(sid, environ):
    """Handle client connection"""
    print(f"Client {sid} connected")
    # Send current state to new connection
    await sio.emit("state_update", optimization_state, room=sid)

@sio.event
async def disconnect(sid):
    """Handle client disconnection"""
    print(f"Client {sid} disconnected")

@sio.event
async def start_optimization(sid, data):
    """Start the optimization process"""
    config = data.get("config", {})
    await handle_start_optimization(config)

@sio.event
async def get_golden_dataset(sid):
    """Get golden dataset"""
    await handle_get_golden_dataset(sid)

@sio.event
async def get_prompt_versions(sid):
    """Get prompt versions"""
    await handle_get_prompt_versions(sid)

@sio.event
async def compare_prompts(sid, data):
    """Compare two prompts"""
    prompt1 = data.get("prompt1")
    prompt2 = data.get("prompt2")
    await handle_compare_prompts(sid, prompt1, prompt2)

async def handle_start_optimization(config: Dict[str, Any]):
    """Start the optimization process with real-time updates"""
    global optimization_state
    
    optimization_state.update({
        "status": "running",
        "progress": 0,
        "session_id": str(uuid.uuid4()),
        "current_step": "Initializing optimization...",
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "results": {}
    })
    
    await sio.emit("optimization_started", optimization_state)
    
    # Run optimization in background
    asyncio.create_task(run_optimization_with_updates(config))

async def run_optimization_with_updates(config: Dict[str, Any]):
    """Run the actual optimization with real-time progress updates"""
    global optimization_state
    
    try:
        # Simulate optimization steps with progress updates
        steps = [
            ("Loading golden dataset...", 10),
            ("Initializing DSPy models...", 20),
            ("Setting up COPRO optimizer...", 30),
            ("Running optimization iterations...", 60),
            ("Evaluating optimized model...", 80),
            ("Saving optimized prompts...", 90),
            ("Generating comparison report...", 100)
        ]
        
        for step, progress in steps:
            optimization_state["current_step"] = step
            optimization_state["progress"] = progress
            
            await sio.emit("progress_update", optimization_state)
            
            # Simulate work (replace with actual optimization)
            await asyncio.sleep(2)
        
        # Mock results (replace with actual optimization results)
        results = {
            "original_score": 3.2,
            "optimized_score": 4.1,
            "improvement": 28.1,
            "optimization_iterations": 15,
            "total_examples": 30,
            "optimized_prompt": "You are SNEHA DIDI, an enhanced healthcare chatbot..."
        }
        
        # Save optimized prompt with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        await save_optimized_prompt(results.get("optimized_prompt", ""), timestamp)
        
        optimization_state.update({
            "status": "completed",
            "progress": 100,
            "current_step": "Optimization completed successfully!",
            "end_time": datetime.now().isoformat(),
            "results": results
        })
        
        await sio.emit("optimization_completed", optimization_state)
        
    except Exception as e:
        optimization_state.update({
            "status": "error",
            "current_step": f"Error: {str(e)}",
            "end_time": datetime.now().isoformat()
        })
        
        await sio.emit("optimization_error", optimization_state)

async def handle_get_golden_dataset(sid: str):
    """Send golden dataset to client"""
    try:
        df = pd.read_csv('data/examples/examples.csv')
        # Clean up the dataframe - remove empty columns
        df = df.dropna(axis=1, how='all')
        
        data = {
            "columns": df.columns.tolist(),
            "rows": df.to_dict('records'),
            "total_count": len(df),
            "filename": "examples.csv"
        }
        
        await sio.emit("golden_dataset", data, room=sid)
        
    except Exception as e:
        await sio.emit("error", {
            "message": f"Failed to load golden dataset: {str(e)}"
        }, room=sid)

async def handle_get_prompt_versions(sid: str):
    """Send available prompt versions to client"""
    try:
        prompts_dir = Path("data/prompts")
        prompts = []
        
        # Get all .md files in prompts directory
        for prompt_file in prompts_dir.glob("*.md"):
            stat = prompt_file.stat()
            prompts.append({
                "filename": prompt_file.name,
                "path": str(prompt_file),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "size": stat.st_size,
                "type": "optimized" if "optimized" in prompt_file.name else "original"
            })
        
        # Sort by modification time (newest first)
        prompts.sort(key=lambda x: x["modified"], reverse=True)
        
        await sio.emit("prompt_versions", prompts, room=sid)
        
    except Exception as e:
        await sio.emit("error", {
            "message": f"Failed to load prompt versions: {str(e)}"
        }, room=sid)

async def handle_compare_prompts(sid: str, prompt1_path: str, prompt2_path: str):
    """Send prompt comparison data to client"""
    try:
        prompts_data = {}
        
        for i, path in enumerate([prompt1_path, prompt2_path], 1):
            if path and Path(path).exists():
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                prompts_data[f"prompt{i}"] = {
                    "path": path,
                    "filename": Path(path).name,
                    "content": content,
                    "line_count": len(content.splitlines()),
                    "char_count": len(content),
                    "modified": datetime.fromtimestamp(Path(path).stat().st_mtime).isoformat()
                }
            else:
                prompts_data[f"prompt{i}"] = None
        
        await sio.emit("prompt_comparison", prompts_data, room=sid)
        
    except Exception as e:
        await sio.emit("error", {
            "message": f"Failed to compare prompts: {str(e)}"
        }, room=sid)

async def save_optimized_prompt(optimized_prompt: str, timestamp: str):
    """Save optimized prompt with timestamp"""
    try:
        prompts_dir = Path("data/prompts")
        prompts_dir.mkdir(exist_ok=True)
        
        filename = f"optimized_{timestamp}.md"
        filepath = prompts_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # Only write the prompt content
            f.write(optimized_prompt if optimized_prompt else "")
        
        print(f"Saved optimized prompt to: {filepath}")
        
    except Exception as e:
        print(f"Failed to save optimized prompt: {e}")

# REST API endpoints for additional functionality
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/status")
async def get_status():
    """Get current optimization status"""
    return optimization_state

@app.get("/api/prompts")
async def list_prompts():
    """List all available prompts"""
    try:
        prompts_dir = Path("data/prompts")
        prompts = []
        
        for prompt_file in prompts_dir.glob("*.md"):
            stat = prompt_file.stat()
            prompts.append({
                "filename": prompt_file.name,
                "path": str(prompt_file),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "size": stat.st_size,
                "type": "optimized" if "optimized" in prompt_file.name else "original"
            })
        
        return {"prompts": prompts}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list prompts: {str(e)}")

@app.get("/api/prompts/{filename}")
async def get_prompt_content(filename: str):
    """Get content of a specific prompt file"""
    try:
        filepath = Path("data/prompts") / filename
        
        if not filepath.exists():
            raise HTTPException(status_code=404, detail="Prompt file not found")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            "filename": filename,
            "content": content,
            "modified": datetime.fromtimestamp(filepath.stat().st_mtime).isoformat(),
            "size": len(content)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read prompt: {str(e)}")

if __name__ == "__main__":
    print("Starting DSPy Optimization Server...")
    print("WebSocket endpoint: ws://localhost:8000/socket.io/")
    print("API endpoint: http://localhost:8000/api/")
    uvicorn.run(socket_app, host="0.0.0.0", port=10000)