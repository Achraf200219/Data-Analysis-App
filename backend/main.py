"""
Enhanced Backend server for multi-format data analysis with file uploads.

This FastAPI application provides:
- File upload and automatic format detection
- Session management for multiple data sources  
- Chat-based natural language querying
- SQL generation for databases and pandas code for tabular data
- Visualization and summary generation
"""

import os
import sys
import tempfile
import uuid
import logging
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from pathlib import Path

# Add the current directory to Python path for importing local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
import aiofiles

import pandas as pd

from openrouter_client import EnhancedOpenRouterClient
from file_handler import detect_and_load_file, get_supported_extensions, FileType, convert_numpy_types
from safe_executor import safe_execute_pandas, safe_validate_sql, SafetyError
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------------------------

# Load environment variables. We try both the backend folder (current file dir)
# and the repository root so users can place .env at either location.
load_dotenv()  # default: current working directory
_here = Path(__file__).resolve().parent
_root_env = _here.parent / ".env"
if _root_env.exists():
    load_dotenv(dotenv_path=_root_env)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models for request/response bodies
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    """Chat message from user"""
    message: str
    source_id: Optional[str] = None

class QueryResponse(BaseModel):
    """Response containing query and results"""
    query_type: str  # "sql" or "pandas"
    query: str
    data_source: str
    source_type: str
    results: List[Dict[str, Any]]
    total_rows: int

class FileInfo(BaseModel):
    """Information about an uploaded file"""
    file_id: str
    name: str
    type: str
    size: int
    metadata: Dict[str, Any]
    schema: Dict[str, Any]

class DataSourceInfo(BaseModel):
    """Information about loaded data sources"""
    sources: Dict[str, FileInfo]
    active_source: Optional[str]

class PlotRequest(BaseModel):
    """Request body for generating a chart and Plotly code."""
    question: str
    query_info: Dict[str, Any]
    df: List[Dict[str, Any]]
    source_id: Optional[str] = None

class SummaryRequest(BaseModel):
    """Request body for generating a summary from a DataFrame."""
    question: str
    df: List[Dict[str, Any]]
    query_info: Optional[Dict[str, Any]] = None
    source_id: Optional[str] = None

class FollowupRequest(BaseModel):
    """Request body for generating follow‑up questions."""
    question: str
    query_info: Dict[str, Any]
    df: List[Dict[str, Any]]
    source_id: Optional[str] = None

class ChartCheckRequest(BaseModel):
    """Request body for determining whether a chart should be generated."""
    df: List[Dict[str, Any]]
    source_id: Optional[str] = None

# ---------------------------------------------------------------------------
# Application Setup
# ---------------------------------------------------------------------------

# Create a single OpenRouter client instance at startup and reuse it for all requests.
client: Optional[EnhancedOpenRouterClient] = None

# Temporary file storage for uploads
UPLOAD_DIRECTORY = Path(tempfile.gettempdir()) / "data_analysis_uploads"
UPLOAD_DIRECTORY.mkdir(exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events."""
    global client
    # Startup
    client = _create_client()
    yield
    # Shutdown
    if client is not None:
        try:
            await client.close()
        except Exception:
            pass

app = FastAPI(
    title="Data Analysis App API", 
    version="1.0.0",
    description="Upload and analyze multiple data formats with natural language queries",
    lifespan=lifespan
)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={"detail": f"Validation error: {str(exc)}"}
    )

# ---------------------------------------------------------------------------
# Client Setup
# ---------------------------------------------------------------------------

def _create_client() -> EnhancedOpenRouterClient:
    """Instantiate and return an OpenRouter client."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is not set. "
            "Please set it to your OpenRouter API key to run the backend."
        )
    model = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o")
    return EnhancedOpenRouterClient(api_key=api_key, model=model)

def get_client() -> EnhancedOpenRouterClient:
    """Dependency to get the OpenRouter client"""
    if client is None:
        raise HTTPException(status_code=500, detail="Client not initialized")
    return client

# ---------------------------------------------------------------------------
# File Management Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/supported-formats")
async def get_supported_formats() -> Dict[str, List[str]]:
    """Get list of supported file formats"""
    extensions = get_supported_extensions()
    
    formats = {
        "database": [".sqlite", ".sqlite3", ".db", ".duckdb", ".mdb", ".accdb", ".sql"],
        "tabular": [".csv", ".xls", ".xlsx", ".parquet"],
        "all": extensions
    }
    
    return formats

@app.post("/api/upload-file", response_model=FileInfo)
async def upload_file(
    file: UploadFile = File(...),
    client: EnhancedOpenRouterClient = Depends(get_client)
) -> FileInfo:
    """Upload and process a data file"""
    
    # Generate unique file ID
    file_id = str(uuid.uuid4())
    
    try:
        # Save uploaded file temporarily
        file_path = UPLOAD_DIRECTORY / f"{file_id}_{file.filename}"
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Detect and load the file
        data_source = detect_and_load_file(str(file_path), file.filename)
        
        # Add to client session
        client.add_data_source(file_id, data_source)
        
        # Get schema and metadata
        schema = data_source.get_schema()
        
        file_info = FileInfo(
            file_id=file_id,
            name=data_source.name,
            type=data_source.file_type.value,
            size=len(content),
            metadata=data_source.metadata,
            schema=schema
        )
        
        return file_info
        
    except Exception as e:
        # Clean up file if processing failed
        if file_path.exists():
            file_path.unlink()
        
        raise HTTPException(status_code=400, detail=f"Failed to process file: {str(e)}")

@app.get("/api/data-sources", response_model=DataSourceInfo)
async def get_data_sources(client: EnhancedOpenRouterClient = Depends(get_client)) -> DataSourceInfo:
    """Get information about all loaded data sources"""
    sources_data = client.get_data_sources()
    
    sources = {}
    for source_id, info in sources_data.items():
        sources[source_id] = FileInfo(
            file_id=source_id,
            name=info["name"],
            type=info["type"],
            size=0,  # Size not tracked in session
            metadata=info["metadata"],
            schema=info["schema"]
        )
    
    return DataSourceInfo(
        sources=sources,
        active_source=client.active_source
    )

@app.post("/api/set-active-source")
async def set_active_source(
    source_id: str = Form(...),
    client: EnhancedOpenRouterClient = Depends(get_client)
) -> Dict[str, str]:
    """Set the active data source"""
    try:
        client.set_active_source(source_id)
        return {"status": "success", "active_source": source_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.delete("/api/remove-source/{source_id}")
async def remove_source(
    source_id: str,
    client: EnhancedOpenRouterClient = Depends(get_client)
) -> Dict[str, str]:
    """Remove a data source"""
    try:
        client.remove_data_source(source_id)
        
        # Clean up temporary file
        for file_path in UPLOAD_DIRECTORY.glob(f"{source_id}_*"):
            try:
                file_path.unlink()
            except Exception:
                pass
        
        return {"status": "success", "removed_source": source_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove source: {str(e)}")

# ---------------------------------------------------------------------------
# Query and Analysis Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/generate-questions")
async def generate_questions(
    source_id: Optional[str] = None,
    client: EnhancedOpenRouterClient = Depends(get_client)
) -> Dict[str, Any]:
    """Generate sample questions for the active or specified data source"""
    try:
        questions: List[str] = await client.generate_questions(source_id)
        return {"questions": questions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate questions: {str(e)}")

@app.post("/api/chat", response_model=QueryResponse)
async def chat_with_data(
    message: ChatMessage,
    client: EnhancedOpenRouterClient = Depends(get_client)
) -> QueryResponse:
    """Process a natural language query and return results"""
    
    if not message.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")
    
    try:
        # Sanitize input
        from safe_executor import sanitize_user_input
        sanitized_message = sanitize_user_input(message.message)
        
        # Generate appropriate query based on data source type
        query_info = await client.generate_query(sanitized_message, message.source_id)
        
        # Execute the query (now with built-in safety checks)
        df = client.execute_query(query_info, message.source_id)
        
        # Limit results for API response (show first 50 rows)
        display_df = df.head(50)
        results = convert_numpy_types(display_df.to_dict(orient="records"))
        
        return QueryResponse(
            query_type=query_info["query_type"],
            query=query_info["query"],
            data_source=query_info["data_source"],
            source_type=query_info["source_type"],
            results=results,
            total_rows=convert_numpy_types(len(df))
        )
        
    except SafetyError as e:
        logger.warning(f"Safety violation in query: {e}")
        raise HTTPException(status_code=400, detail=f"Query rejected for safety reasons: {str(e)}")
    except ValueError as e:
        logger.warning(f"Query validation error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid query: {str(e)}")
    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")

@app.post("/api/should-generate-chart")
async def should_generate_chart(
    req: ChartCheckRequest,
    client: EnhancedOpenRouterClient = Depends(get_client)
) -> Dict[str, bool]:
    """Check whether a chart should be generated for the given DataFrame"""
    df = pd.DataFrame(req.df)
    try:
        flag: bool = await client.should_generate_chart(df, req.source_id)
        return {"should_generate_chart": flag}
    except Exception:
        return {"should_generate_chart": False}

@app.post("/api/generate-plot")
async def generate_plot(
    req: PlotRequest,
    client: EnhancedOpenRouterClient = Depends(get_client)
) -> Dict[str, Any]:
    """Generate Plotly code and figure JSON for a question/query/dataframe combo"""
    df = pd.DataFrame(req.df)
    try:
        code: str = await client.generate_plotly_code(
            req.question, req.query_info, df, req.source_id
        )
    except Exception as e:
        return {"plotly_code": "", "fig": None, "error": str(e)}
    
    fig_json: Optional[str] = None
    if code and code.strip() != "":
        try:
            fig = client.get_plotly_figure(code, df, req.source_id)
            fig_json = fig.to_json()
        except Exception:
            fig_json = None
    
    return {"plotly_code": code, "fig": fig_json}

@app.post("/api/generate-summary")
async def generate_summary(
    req: SummaryRequest,
    client: EnhancedOpenRouterClient = Depends(get_client)
) -> Dict[str, Any]:
    """Generate a textual summary of the query result"""
    df = pd.DataFrame(req.df)
    try:
        summary: Optional[str] = await client.generate_summary(
            req.question, df, req.query_info, req.source_id
        )
        return {"summary": summary}
    except Exception:
        return {"summary": None}

@app.post("/api/generate-followup")
async def generate_followup(
    req: FollowupRequest,
    client: EnhancedOpenRouterClient = Depends(get_client)
) -> Dict[str, Any]:
    """Generate follow‑up questions based on the original question, query and DataFrame"""
    df = pd.DataFrame(req.df)
    try:
        followups: List[str] = await client.generate_followup_questions(
            req.question, req.query_info, df, req.source_id
        )
        return {"followup_questions": followups}
    except Exception:
        return {"followup_questions": []}

# ---------------------------------------------------------------------------
# Legacy Compatibility Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/generate-sql")
async def generate_sql_legacy(
    question: str,
    source_id: Optional[str] = None,
    client: EnhancedOpenRouterClient = Depends(get_client)
) -> Dict[str, str]:
    """Legacy endpoint - Generate SQL for a natural language question"""
    if not question:
        raise HTTPException(status_code=400, detail="No question provided")
    
    try:
        sql: str = await client.generate_sql(question, source_id=source_id)
        return {"sql": sql}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate SQL: {str(e)}")

# ---------------------------------------------------------------------------
# Health and Info Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Data Analysis App API",
        "version": "1.0.0",
        "description": "Upload and analyze multiple data formats with natural language queries",
        "supported_formats": get_supported_extensions(),
        "endpoints": {
            "upload": "/api/upload-file",
            "chat": "/api/chat",
            "sources": "/api/data-sources",
            "questions": "/api/generate-questions"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "client_initialized": client is not None,
        "upload_directory": str(UPLOAD_DIRECTORY),
        "supported_extensions": len(get_supported_extensions())
    }

@app.get("/api/rate-limit-status")
async def get_rate_limit_status():
    """Get OpenRouter API rate limit status"""
    if not client:
        raise HTTPException(status_code=503, detail="Client not initialized")
    
    try:
        status = await client.get_rate_limit_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get rate limit status: {e}")
        return {
            "status": "error",
            "error": str(e),
            "message": "Unable to retrieve rate limit information"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)