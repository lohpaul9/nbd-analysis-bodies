from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import json
import numpy as np
from data_reader import *
from models import run_experiments as run_model_experiments
from json_tricks import dumps, loads
import os

app = FastAPI()

# Mount the static files from the frontend build
app.mount("/assets", StaticFiles(directory="../frontend/dist/assets"), name="assets")

class ExperimentConfig(BaseModel):
    name: str
    filters: Dict[str, Any]

class ExperimentsRequest(BaseModel):
    experiments: List[ExperimentConfig]

@app.post("/api/run-experiments")
async def run_experiments(request: ExperimentsRequest):
    try:
        # Convert the request data into the format expected by run_experiments
        experiment_tuples = [
            (exp.name, exp.filters) for exp in request.experiments
        ]
        print(experiment_tuples)
        
        # Run the experiments
        results = run_model_experiments(experiment_tuples)
        
        # Convert numpy arrays to lists for JSON serialization
        processed_results = []
        for result in results:
            # serialized_results = loads(dumps(result))
            processed_results.append(result)
            
        return processed_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/filter-options")
async def get_filter_options():
    # Return all available filter options
    filters_with_options = dict()
    for filter in FILTER_LOOKUP_MAP:
        filters_with_options[filter] = FILTER_LOOKUP_MAP[filter][0]
    
    return filters_with_options

# Serve the frontend
@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    # Serve index.html for all routes except /api
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not found")
    
    frontend_path = "../frontend/dist"
    file_path = os.path.join(frontend_path, full_path)
    
    # If the specific file exists, serve it
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path)
    
    # Otherwise serve index.html
    return FileResponse(os.path.join(frontend_path, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)