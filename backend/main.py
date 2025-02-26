from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import json
import numpy as np
from data_reader import *
from models import run_experiments as run_model_experiments
from json_tricks import dumps, loads

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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