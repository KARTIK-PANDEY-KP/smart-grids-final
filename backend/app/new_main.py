import os
import json
from dotenv import load_dotenv
import httpx
import asyncio
import re

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pinecone import Pinecone

#main other than main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import os
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
import openai
import logging
import time
import sqlite3
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ─── LOAD .env ───────────────────────────────────────────────
load_dotenv()

# Create data directory if it doesn't exist
data_dir = Path("./data")
data_dir.mkdir(exist_ok=True)


# Database setup
DB_PATH = data_dir / "spreadsheet.db"
def get_db_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

# Initialize database
def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    # Create tables table to store sheet info
    cur.execute('''
    CREATE TABLE IF NOT EXISTS sheets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create rows table to store the actual data
    cur.execute('''
    CREATE TABLE IF NOT EXISTS sheet_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sheet_id INTEGER NOT NULL,
        row_index INTEGER NOT NULL,
        column_name TEXT NOT NULL,
        value TEXT,
        FOREIGN KEY (sheet_id) REFERENCES sheets (id),
        UNIQUE(sheet_id, row_index, column_name)
    )
    ''')
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {DB_PATH}")

# Initialize the database at startup
init_db()

# Check if API key is set
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("OPENAI_API_KEY environment variable is not set")
    raise ValueError("OPENAI_API_KEY environment variable is not set")
else:
    logger.info("OPENAI_API_KEY is set (length: %d)", len(api_key))


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-west1-gcp")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"
TABLE_API_URL = os.getenv("TABLE_API_URL", "http://localhost:8000/api/table/json")

# Check if keys are available
if not PERPLEXITY_API_KEY:
    print("WARNING: PERPLEXITY_API_KEY is not set. Web search functionality will not work.")
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY is not set. API calls to OpenAI will fail.")
if not PINECONE_API_KEY:
    print("WARNING: PINECONE_API_KEY is not set. Interview search functionality will not work.")

async def fetch_table_data():
    """Fetch current table data from the API"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            print(f"Fetching table data from {TABLE_API_URL}")
            response = await client.get(TABLE_API_URL)
            
            if response.status_code == 200:
                data = response.json()
                if "data" in data and "columns" in data:
                    print("Successfully fetched table data")
                    # Convert to expected format with tableData
                    return {"tableData": data["data"]}
                else:
                    print("Warning: Response did not contain expected data structure")
                    return None
            else:
                print(f"Error fetching table data: Status {response.status_code}")
                return None
    except Exception as e:
        print(f"Exception while fetching table data: {str(e)}")
        return None

def create_global_system_prompt():
    """Create the global system prompt that will be used for all requests"""
    base_prompt = "" # enter anything here that needs to be globally sent to all the OpenAI calls
    
    # This function will be called synchronously but we'll have the table data cached
    # so it's not an issue
    return base_prompt

_cached_table_data = None
_last_table_fetch_time = 0

async def get_system_prompt_with_table_data():
    """Get the system prompt, optionally with table data if available"""
    global _cached_table_data, _last_table_fetch_time
    
    # Check if we need to refresh the cache (every 60 seconds)
    current_time = asyncio.get_event_loop().time()
    if current_time - _last_table_fetch_time > 60 or _cached_table_data is None:
        _cached_table_data = await fetch_table_data()
        _last_table_fetch_time = current_time
    
    base_prompt = ""
    
    # If we have table data, add it to the prompt
    if _cached_table_data and "tableData" in _cached_table_data:
        table_data = _cached_table_data["tableData"]
        if table_data and len(table_data) > 0:
            # Create a markdown table representation
            table_str = "Here is the current table data you can reference:\n\n"
            
            # Add headers
            if table_data and len(table_data) > 0:
                headers = table_data[0].keys()
                table_str += "| " + " | ".join(headers) + " |\n"
                table_str += "| " + " | ".join(["---" for _ in headers]) + " |\n"
                
                # Add rows
                for row in table_data:
                    table_str += "| " + " | ".join([str(row.get(h, "")) for h in headers]) + " |\n"
            
            # Add the table data to the prompt
            return f"{base_prompt}\n\n{table_str}\n\nRefer to this table data when answering questions about student information."
    
    # Return the base prompt if no table data
    return base_prompt

# Initialize Pinecone client
try:
    print(f"Initializing Pinecone with API key (first 8 chars): {PINECONE_API_KEY[:8]}... and environment: {PINECONE_ENV}")
    pc = Pinecone(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )
    
    print("Pinecone client initialized. Attempting to connect to index 'cq-transcripts-1'...")
    
    pinecone_index = pc.Index(
        name="cq-transcripts-1",
        pool_threads=50,
        connection_pool_maxsize=50,
    )
    
    # Test the connection by listing namespaces (collections)
    try:
        print("Testing Pinecone connection...")
        # Attempt to do a simple operation to verify the connection works
        describe_response = pinecone_index.describe_index_stats()
        print(f"Pinecone connection successful. Index stats: {describe_response}")
        
        # Get a list of namespaces
        namespaces = describe_response.get("namespaces", {})
        print(f"Available namespaces: {list(namespaces.keys())}")
        print("Pinecone initialized successfully.")
    except Exception as e:
        print(f"WARNING: Pinecone connection test failed: {str(e)}")
        # Still continue as the index might be valid
except Exception as e:
    print(f"ERROR initializing Pinecone: {str(e)}")
    pinecone_index = None

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten this in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
openai_client = openai.AsyncOpenAI(api_key=api_key)

# Store for last enriched table data (for demo purposes)
last_enriched_table = {
    "data": [],
    "timestamp": None
}

# Current table data (in-memory)
current_table = {
    "data": [],
    "timestamp": time.time()
}

class EnrichmentRequest(BaseModel):
    rowData: Dict[str, str]
    prompt: str
    columnName: str

class EnrichmentResponse(BaseModel):
    result: str
    confidence: Optional[float] = None
    processingTime: Optional[float] = None

class BatchEnrichmentRequest(BaseModel):
    rows: List[Dict[str, Any]]
    prompt: str
    columnName: str

class BatchEnrichmentResponse(BaseModel):
    results: List[Dict[str, Any]]
    processingTime: Optional[float] = None

class EnrichedTableResponse(BaseModel):
    tableData: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    message: str

class CurrentTableResponse(BaseModel):
    tableData: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    message: str = "Current table data returned successfully"

class UpdateTableRequest(BaseModel):
    tableData: List[Dict[str, Any]]

class DatabaseSaveResponse(BaseModel):
    success: bool
    message: str
    sheet_name: str
    row_count: int
    column_count: int

class ColumnValuesResponse(BaseModel):
    columns: List[str]
    values: Dict[str, List[str]]
    message: str = "Column values returned successfully"

class TableJsonResponse(BaseModel):
    columns: List[str]
    data: List[Dict[str, str]]
    message: str = "Table data returned as JSON"

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def get_ai_completion(prompt: str) -> str:
    try:
        logger.info("Attempting to get AI completion for prompt: %s", prompt[:50] + "...")
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a data analysis assistant. Provide concise answers and exactly what "
                        "the user asks for. For example, if user says add a value to a column then "
                        "only add the value and print the value with no additional information or "
                        "English. Simply analyze the query and identify what the user is asking since "
                        "it's a column for a spreadsheet assume what forms of simplification you may "
                        "have to do. IF THE RELEVANT DATA NEEDED FOR MAKING THE VALUE IS NOT PRESENT "
                        "DON'T ASSUME AND KEEP IT EMPTY. Keep the output fact checked, only use what "
                        "is in the input. For example, if asked to extract company from email "
                        "'john@acme.com', only extract 'acme' without adding corp/com/co/llc etc."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=50
        )
        result = response.choices[0].message.content.strip()
        logger.info("Successfully got AI completion")
        return result
    except Exception as e:
        logger.error("Unexpected Error: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/api/enrich", response_model=EnrichmentResponse)
async def enrich_data(request: EnrichmentRequest):
    start_time = time.time()
    try:
        # Format the prompt with the row data
        formatted_prompt = f"{request.prompt}\n\nData:\n"
        for key, value in request.rowData.items():
            formatted_prompt += f"{key}: {value}\n"
        
        # Get AI completion
        result = await get_ai_completion(formatted_prompt)
        
        # Store the enriched row for the GET endpoint
        enriched_row = request.rowData.copy()
        enriched_row[request.columnName] = result
        
        global last_enriched_table
        last_enriched_table["data"] = [enriched_row]
        last_enriched_table["timestamp"] = time.time()
        
        processing_time = time.time() - start_time
        
        return EnrichmentResponse(
            result=result,
            confidence=0.95,  # Mock confidence score
            processingTime=processing_time
        )
    
    except Exception as e:
        logger.error("Error in enrich_data: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/enrich/batch", response_model=BatchEnrichmentResponse)
async def enrich_batch(request: BatchEnrichmentRequest):
    start_time = time.time()
    try:
        results = []
        enriched_rows = []
        
        for row in request.rows:
            # Format the prompt with the row data
            formatted_prompt = f"{request.prompt}\n\nData:\n"
            for key, value in row["data"].items():
                formatted_prompt += f"{key}: {value}\n"
            
            # Get AI completion
            result = await get_ai_completion(formatted_prompt)
            
            # Store for the GET endpoint
            enriched_row = row["data"].copy()
            enriched_row[request.columnName] = result
            enriched_row["rowId"] = row["rowId"]
            enriched_rows.append(enriched_row)
            
            results.append({
                "rowId": row["rowId"],
                "result": result
            })
        
        # Update last_enriched_table for the GET endpoint
        global last_enriched_table
        last_enriched_table["data"] = enriched_rows
        last_enriched_table["timestamp"] = time.time()
        
        processing_time = time.time() - start_time
        
        return BatchEnrichmentResponse(
            results=results,
            processingTime=processing_time
        )
    
    except Exception as e:
        logger.error("Error in enrich_batch: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/table/enriched", response_model=EnrichedTableResponse)
async def get_enriched_table():
    try:
        # If no data has been processed yet, return sample data
        if not last_enriched_table["data"]:
            sample_data = [
                {"id": 1, "name": "John Doe", "email": "john@example.com", "company": "Example Inc", "enriched_field": "example"},
                {"id": 2, "name": "Jane Smith", "email": "jane@acme.org", "company": "Acme", "enriched_field": "acme"},
                {"id": 3, "name": "Bob Johnson", "email": "bob@company.net", "company": "Company", "enriched_field": "company"}
            ]
            
            return EnrichedTableResponse(
                tableData=sample_data,
                metadata={
                    "source": "sample_data",
                    "rowCount": len(sample_data),
                    "timestamp": time.time()
                },
                message="Sample data returned, no enrichment has been performed yet"
            )
        
        # Return the most recently enriched data
        return EnrichedTableResponse(
            tableData=last_enriched_table["data"],
            metadata={
                "source": "last_enriched",
                "timestamp": last_enriched_table["timestamp"],
                "rowCount": len(last_enriched_table["data"])
            },
            message="Last enriched table data returned successfully"
        )
        
    except Exception as e:
        logger.error("Error in get_enriched_table: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/table/current", response_model=CurrentTableResponse)
async def get_current_table():
    """
    Returns the current state of the table, always reflecting the most up-to-date data.
    This endpoint can be called at any time to get the latest table state.
    """
    try:
        # Make sure we have the latest table data with all columns
        logger.info(f"Returning current table with {len(current_table['data'])} rows")
        
        return CurrentTableResponse(
            tableData=current_table["data"],
            metadata={
                "timestamp": current_table["timestamp"],
                "rowCount": len(current_table["data"]),
                "columnCount": len(current_table["data"][0]) if current_table["data"] else 0,
                "columns": list(current_table["data"][0].keys()) if current_table["data"] else []
            }
        )
    except Exception as e:
        logger.error(f"Error in get_current_table: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/table/update", response_model=CurrentTableResponse)
async def update_table(request: UpdateTableRequest):
    """
    Updates the current table with new data from the frontend.
    This allows the frontend to save its table state to the backend.
    """
    try:
        # Update the current table with the provided data
        global current_table
        
        # Log information about the update
        old_columns = set(current_table["data"][0].keys()) if current_table["data"] else set()
        new_columns = set(request.tableData[0].keys()) if request.tableData else set()
        added_columns = new_columns - old_columns
        
        logger.info(f"Updating table: {len(request.tableData)} rows, {len(new_columns)} columns")
        if added_columns:
            logger.info(f"New columns detected: {added_columns}")
        
        # Save the complete table data
        current_table["data"] = request.tableData
        current_table["timestamp"] = time.time()
        
        return CurrentTableResponse(
            tableData=current_table["data"],
            metadata={
                "timestamp": current_table["timestamp"],
                "rowCount": len(current_table["data"]),
                "columnCount": len(current_table["data"][0]) if current_table["data"] else 0,
                "columns": list(current_table["data"][0].keys()) if current_table["data"] else [],
                "status": "updated",
                "added_columns": list(added_columns) if added_columns else []
            },
            message="Table updated successfully"
        )
    except Exception as e:
        logger.error(f"Error in update_table: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/table/sync", response_model=CurrentTableResponse)
async def sync_table(request: UpdateTableRequest):
    """
    Sync the table data from the frontend to the backend.
    This should be called whenever the frontend table changes.
    """
    try:
        global current_table
        
        # Compare what's changed
        old_data = current_table["data"]
        new_data = request.tableData
        
        # Log information about the sync
        old_rows = len(old_data)
        new_rows = len(new_data)
        old_columns = set()
        new_columns = set()
        
        if old_data and old_rows > 0:
            old_columns = set(old_data[0].keys())
        
        if new_data and new_rows > 0:
            new_columns = set(new_data[0].keys())
        
        added_columns = new_columns - old_columns
        removed_columns = old_columns - new_columns
        row_diff = new_rows - old_rows
        
        # Log detailed changes for debugging
        logger.info(f"Table sync: {new_rows} rows ({row_diff:+d}), {len(new_columns)} columns")
        if added_columns:
            logger.info(f"New columns: {added_columns}")
        if removed_columns:
            logger.info(f"Removed columns: {removed_columns}")
        
        # Update the table data
        current_table["data"] = new_data
        current_table["timestamp"] = time.time()
        
        # Return the updated table data
        return CurrentTableResponse(
            tableData=current_table["data"],
            metadata={
                "timestamp": current_table["timestamp"],
                "rowCount": new_rows,
                "columnCount": len(new_columns),
                "columns": list(new_columns),
                "changes": {
                    "rows": row_diff,
                    "added_columns": list(added_columns),
                    "removed_columns": list(removed_columns)
                }
            },
            message=f"Table synced: {new_rows} rows, {len(new_columns)} columns"
        )
    except Exception as e:
        logger.error(f"Error syncing table: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/table/save-to-database", response_model=DatabaseSaveResponse)
async def save_to_database(request: UpdateTableRequest):
    """
    Permanently store the current spreadsheet data in the SQLite database.
    This endpoint overwrites any existing data for the active sheet.
    """
    try:
        # Get the data to store
        data = request.tableData
        
        if not data:
            raise HTTPException(status_code=400, detail="No data provided to save")
        
        # For simplicity, we'll use a fixed sheet name
        sheet_name = "main_sheet"
        
        # Get column names from the first row
        if not data[0]:
            raise HTTPException(status_code=400, detail="Data format is invalid")
            
        columns = list(data[0].keys())
        
        conn = get_db_connection()
        try:
            # Begin transaction
            conn.execute("BEGIN TRANSACTION")
            
            # Check if sheet exists
            cur = conn.cursor()
            cur.execute("SELECT id FROM sheets WHERE name = ?", (sheet_name,))
            sheet_row = cur.fetchone()
            
            if sheet_row:
                # Sheet exists, update it
                sheet_id = sheet_row['id']
                # Update timestamp
                cur.execute("UPDATE sheets SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (sheet_id,))
                
                # Delete existing data for this sheet
                cur.execute("DELETE FROM sheet_data WHERE sheet_id = ?", (sheet_id,))
            else:
                # Create new sheet
                cur.execute("INSERT INTO sheets (name) VALUES (?)", (sheet_name,))
                sheet_id = cur.lastrowid
            
            # Insert all the data
            for row_idx, row_data in enumerate(data):
                for col_name, value in row_data.items():
                    cur.execute(
                        "INSERT INTO sheet_data (sheet_id, row_index, column_name, value) VALUES (?, ?, ?, ?)",
                        (sheet_id, row_idx, col_name, str(value))
                    )
            
            # Commit transaction
            conn.commit()
            
            logger.info(f"Successfully saved sheet '{sheet_name}' to database with {len(data)} rows and {len(columns)} columns")
            
            return DatabaseSaveResponse(
                success=True,
                message=f"Successfully saved to database",
                sheet_name=sheet_name,
                row_count=len(data),
                column_count=len(columns)
            )
            
        except Exception as e:
            # Rollback on error
            conn.rollback()
            logger.error(f"Database error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        finally:
            conn.close()
            
    except Exception as e:
        logger.error(f"Error saving to database: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/table/load-from-database", response_model=CurrentTableResponse)
async def load_from_database():
    """
    Load the spreadsheet data from the database.
    """
    try:
        # For simplicity, we'll use a fixed sheet name
        sheet_name = "main_sheet"
        
        conn = get_db_connection()
        try:
            # Get sheet id
            cur = conn.cursor()
            cur.execute("SELECT id FROM sheets WHERE name = ?", (sheet_name,))
            sheet_row = cur.fetchone()
            
            if not sheet_row:
                # No saved data
                return CurrentTableResponse(
                    tableData=[],
                    metadata={
                        "timestamp": time.time(),
                        "rowCount": 0,
                        "columnCount": 0,
                        "columns": []
                    },
                    message="No saved data found in database"
                )
            
            sheet_id = sheet_row['id']
            
            # First, get distinct column names in order of their first appearance
            cur.execute("""
                SELECT DISTINCT column_name 
                FROM sheet_data 
                WHERE sheet_id = ? 
                GROUP BY column_name
                ORDER BY MIN(id)
            """, (sheet_id,))
            
            ordered_columns = [row['column_name'] for row in cur.fetchall()]
            
            # Get all data for this sheet
            cur.execute("""
                SELECT row_index, column_name, value 
                FROM sheet_data 
                WHERE sheet_id = ?
                ORDER BY row_index, column_name
            """, (sheet_id,))
            
            all_data = cur.fetchall()
            
            # Process the data into rows
            rows_dict = {}
            
            for row in all_data:
                row_idx = row['row_index']
                col_name = row['column_name']
                value = row['value']
                
                if row_idx not in rows_dict:
                    rows_dict[row_idx] = {}
                
                rows_dict[row_idx][col_name] = value
            
            # Convert to ordered list, ensuring consistent column order
            table_data = []
            for row_idx in sorted(rows_dict.keys()):
                row_data = {}
                # Add columns in the correct order
                for col_name in ordered_columns:
                    row_data[col_name] = rows_dict[row_idx].get(col_name, "")
                table_data.append(row_data)
            
            # Update the current table in memory
            global current_table
            current_table["data"] = table_data
            current_table["timestamp"] = time.time()
            
            logger.info(f"Loaded {len(table_data)} rows with {len(ordered_columns)} columns from database")
            
            return CurrentTableResponse(
                tableData=table_data,
                metadata={
                    "timestamp": time.time(),
                    "rowCount": len(table_data),
                    "columnCount": len(ordered_columns),
                    "columns": ordered_columns,
                    "source": "database"
                },
                message=f"Successfully loaded {len(table_data)} rows from database"
            )
            
        finally:
            conn.close()
            
    except Exception as e:
        logger.error(f"Error loading from database: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/table/columns", response_model=ColumnValuesResponse)
async def get_column_values():
    """
    Returns all column names and their unique values from the current table.
    Columns are returned in their original order.
    """
    try:
        # Check if we have any data
        if not current_table["data"] or len(current_table["data"]) == 0:
            # Try to load from database
            conn = get_db_connection()
            try:
                sheet_name = "main_sheet"
                cur = conn.cursor()
                
                # Check if sheet exists
                cur.execute("SELECT id FROM sheets WHERE name = ?", (sheet_name,))
                sheet_row = cur.fetchone()
                
                if not sheet_row:
                    # No data in database either
                    return ColumnValuesResponse(
                        columns=[],
                        values={},
                        message="No data available"
                    )
                
                sheet_id = sheet_row['id']
                
                # Get distinct column names in order
                cur.execute("""
                    SELECT DISTINCT column_name 
                    FROM sheet_data 
                    WHERE sheet_id = ? 
                    GROUP BY column_name
                    ORDER BY MIN(id)
                """, (sheet_id,))
                
                columns = [row['column_name'] for row in cur.fetchall()]
                
                # Get all values for each column
                values = {}
                for column in columns:
                    cur.execute("""
                        SELECT DISTINCT value
                        FROM sheet_data
                        WHERE sheet_id = ? AND column_name = ?
                        ORDER BY value
                    """, (sheet_id, column))
                    
                    values[column] = [row['value'] for row in cur.fetchall() if row['value']]
                
                return ColumnValuesResponse(
                    columns=columns,
                    values=values,
                    message="Column values loaded from database"
                )
            finally:
                conn.close()
        
        # If we have in-memory data, use that
        data = current_table["data"]
        
        # Extract columns in order (from first row)
        columns = list(data[0].keys()) if data else []
        
        # Get unique values for each column
        values = {}
        for column in columns:
            # Extract all values for this column and filter out empty ones
            column_values = [row[column] for row in data if row.get(column)]
            # Remove duplicates and sort
            unique_values = sorted(list(set(column_values)))
            values[column] = unique_values
        
        return ColumnValuesResponse(
            columns=columns,
            values=values,
            message=f"Returned values for {len(columns)} columns"
        )
    
    except Exception as e:
        logger.error(f"Error getting column values: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/table/json", response_model=TableJsonResponse)
async def get_table_json():
    """
    Returns the entire table as JSON with columns in their original order.
    """
    try:
        # Check if we have any data in memory
        if not current_table["data"] or len(current_table["data"]) == 0:
            # Try to load from database
            conn = get_db_connection()
            try:
                sheet_name = "main_sheet"
                cur = conn.cursor()
                
                # Check if sheet exists
                cur.execute("SELECT id FROM sheets WHERE name = ?", (sheet_name,))
                sheet_row = cur.fetchone()
                
                if not sheet_row:
                    # No data in database either
                    return TableJsonResponse(
                        columns=[],
                        data=[],
                        message="No data available"
                    )
                
                sheet_id = sheet_row['id']
                
                # Get distinct column names in order
                cur.execute("""
                    SELECT DISTINCT column_name 
                    FROM sheet_data 
                    WHERE sheet_id = ? 
                    GROUP BY column_name
                    ORDER BY MIN(id)
                """, (sheet_id,))
                
                columns = [row['column_name'] for row in cur.fetchall()]
                
                # Get all data organized by row
                cur.execute("""
                    SELECT row_index, column_name, value 
                    FROM sheet_data 
                    WHERE sheet_id = ?
                    ORDER BY row_index
                """, (sheet_id,))
                
                all_data = cur.fetchall()
                
                # Group by row index
                rows_dict = {}
                for row in all_data:
                    row_idx = row['row_index']
                    col_name = row['column_name']
                    value = row['value']
                    
                    if row_idx not in rows_dict:
                        rows_dict[row_idx] = {}
                    
                    rows_dict[row_idx][col_name] = value
                
                # Convert to list of ordered dicts to preserve column order
                table_data = []
                for row_idx in sorted(rows_dict.keys()):
                    row_data = {}
                    for col in columns:
                        row_data[col] = rows_dict[row_idx].get(col, "")
                    table_data.append(row_data)
                
                return TableJsonResponse(
                    columns=columns,
                    data=table_data,
                    message=f"Loaded table with {len(table_data)} rows and {len(columns)} columns from database"
                )
            finally:
                conn.close()
        
        # If we have in-memory data, use that
        data = current_table["data"]
        
        # Extract columns in order (from first row)
        columns = list(data[0].keys()) if data else []
        
        # Return all rows with columns in order
        return TableJsonResponse(
            columns=columns,
            data=data,
            message=f"Returned table with {len(data)} rows and {len(columns)} columns"
        )
    
    except Exception as e:
        logger.error(f"Error getting table as JSON: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def web_search(query, stream_callback=None):
    """Perform a web search using Perplexity API with optional streaming."""
    print(f"Starting web search for query: '{query}'")
    
    # Check if PERPLEXITY_API_KEY is available
    if not PERPLEXITY_API_KEY:
        print("PERPLEXITY_API_KEY is not set. Using fallback search.")
        if stream_callback and callable(stream_callback):
            await stream_callback("Web search API key not found. Using alternative search method...")
        return await fallback_search(query, stream_callback)
    
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }
    
    body = {
        "model": "sonar-reasoning-pro",
        "messages": [{"role": "user", "content": query}],
    }
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:  # Increase timeout to 2 minutes
            print(f"Making Perplexity API request for query: {query}")
            print(f"Using API key (first 8 chars): {PERPLEXITY_API_KEY[:8]}...")
            
            try:
                print("Attempting to connect to Perplexity API...")
                response = await client.post(
                PERPLEXITY_URL,
                headers=headers,
                json=body
                )
                print(f"Perplexity API call completed. Response status: {response.status_code}")
            
                # Dump headers for debugging
                print(f"Response headers: {dict(response.headers)}")
            
                if response.status_code == 200:
                    try:
                        data = response.json()
                        print(f"Response JSON keys: {data.keys()}")
                        
                        # Dump entire response for debugging (truncated)
                        response_dump = str(data)[:1000] + "..." if len(str(data)) > 1000 else str(data)
                        print(f"Response dump: {response_dump}")
                        
                        if "choices" not in data or len(data["choices"]) == 0:
                            error_message = f"Unexpected response format: No choices in response. Full response: {data}"
                            print(error_message)
                            if stream_callback and callable(stream_callback):
                                await stream_callback("Error: Unexpected response format from search API")
                            return error_message
                        
                        result = data["choices"][0]["message"]["content"]
                        print(f"Perplexity search result (truncated): {result[:100]}...")
                        
                        # If streaming callback is provided, send chunks gradually
                        if stream_callback and callable(stream_callback):
                            # Split the result into sentences for more natural streaming
                            sentences = re.split(r'(?<=[.!?])\s+', result)
                    
                            # Stream each sentence with a small delay
                            for sentence in sentences:
                                sentence = sentence.strip()
                                if sentence:
                                    # Further break down long sentences into smaller chunks
                                    chunk_size = 30  # characters per chunk
                                    for i in range(0, len(sentence), chunk_size):
                                        chunk = sentence[i:i+chunk_size]
                                        await stream_callback(chunk)
                                        await asyncio.sleep(0.1)  # Small delay for more natural streaming
                
                        return result
                    except json.JSONDecodeError as e:
                        error_message = f"Error decoding JSON response: {str(e)}. Response content: {response.text[:500]}"
                        print(error_message)
                        if stream_callback and callable(stream_callback):
                            await stream_callback("Error: Invalid response from search API")
                        return error_message
                    except KeyError as e:
                        error_message = f"Key error in response: {str(e)}. Response content: {response.json()}"
                        print(error_message)
                        if stream_callback and callable(stream_callback):
                            await stream_callback(f"Error: Missing data in API response: {str(e)}")
                        return error_message
                else:
                    error_message = f"Error in web search: Status {response.status_code}"
                    if response.status_code == 401:
                        error_message += " - Unauthorized. Check your API key."
                    elif response.status_code == 403:
                        error_message += " - Forbidden. API key may be invalid or expired."
                    else:
                        try:
                            error_message += f", Response: {response.text}"
                        except Exception:
                            error_message += " (Could not read response body)"
                    
                    print(error_message)
                    
                    # If there's an error when streaming, use fallback search
                    print("Web search API failed. Using fallback search.")
                    if stream_callback and callable(stream_callback):
                        await stream_callback("Web search encountered an error. Using alternative search method...")
                    
                    return await fallback_search(query, stream_callback)
            except httpx.TimeoutException:
                error_message = "Search request timed out after 120 seconds"
                print(f"ERROR: {error_message}")
                if stream_callback and callable(stream_callback):
                    await stream_callback("Web search timed out. Using alternative search method...")
                return await fallback_search(query, stream_callback)
            except httpx.RequestError as exc:
                error_message = f"An error occurred while requesting from Perplexity API: {exc}"
                print(f"HTTPX REQUEST ERROR: {error_message}")
                if stream_callback and callable(stream_callback):
                    await stream_callback(f"Web search connection error. Using alternative search method... Error: {str(exc)[:50]}")
                return await fallback_search(query, stream_callback)
            except Exception as e:
                # Catch any other exceptions during the API call or initial processing
                error_message = f"Unexpected exception during Perplexity API call or initial response handling: {str(e)}"
                print(f"UNEXPECTED ERROR (during API call phase): {error_message}")
                if stream_callback and callable(stream_callback):
                    await stream_callback(f"Unexpected web search error. Using alternative search method... Error: {str(e)[:50]}")
                return await fallback_search(query, stream_callback)
                
    except Exception as e:
        error_message = f"Outer exception in web search: {str(e)}" # Renamed for clarity
        print(f"OUTER EXCEPTION (web_search): {error_message}")
        
        # If there's an exception when streaming, use fallback search
        if stream_callback and callable(stream_callback):
            await stream_callback(f"Web search error: {str(e)[:30]}... Using alternative search method...")
        
        return await fallback_search(query, stream_callback)

async def interview_search(query, company_name="innabox", stream_callback=None):
    """Search for interview transcripts using Pinecone."""
    print(f"Starting interview search for query: '{query}' in company: '{company_name}'")
    
    if not pinecone_index:
        error_message = "Pinecone is not initialized. Cannot perform interview search."
        print(error_message)
        
        # If there's an error when streaming, send a simplified error message
        if stream_callback and callable(stream_callback):
            await stream_callback("Error: Pinecone is not initialized")
            
        return error_message
    
    try:
        # Send initial message if streaming
        if stream_callback:
            await stream_callback(f"Searching interview transcripts for: {query}")
            await stream_callback("\nGenerating embeddings...")
        
        # Create embedding for query
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        
        embed_body = {
            "model": "text-embedding-3-small",
            "input": query
        }
        
        print(f"Generating embeddings for query: '{query}'")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            embed_response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json=embed_body
            )
            
            print(f"Embedding API response status: {embed_response.status_code}")
            
            if embed_response.status_code != 200:
                error_message = f"Error generating embeddings: {embed_response.text}"
                print(error_message)
                
                # If there's an error when streaming, send a simplified error message
                if stream_callback and callable(stream_callback):
                    await stream_callback(f"Error: Unable to generate embeddings (status {embed_response.status_code})")
                    
                return error_message
            
            embed_data = embed_response.json()
            vector = embed_data["data"][0]["embedding"]
            
            # Log the vector dimensions to ensure it's valid
            print(f"Generated embedding vector of dimension: {len(vector)}")
            
            # Query Pinecone with the embedding
            if stream_callback:
                await stream_callback("\nSearching Pinecone index...")
            
            print(f"Querying Pinecone index with namespace: '{company_name}'")
            
            # Execute the Pinecone query
            response = pinecone_index.query(
                vector=vector,
                top_k=3,
                include_metadata=True,
                namespace=company_name
            )
            
            # Log the number of matches
            print(f"Pinecone query returned {len(response.matches)} matches")
            
            # Process and format results
            all_results = []
            formatted_results = ""
            
            if len(response.matches) == 0:
                no_results_message = f"No results found for '{query}' in company '{company_name}'"
                print(no_results_message)
                if stream_callback:
                    await stream_callback(f"\n{no_results_message}")
                return no_results_message
            
            for i, match in enumerate(response.matches):
                meta = match.metadata or {}
                
                # Pick the first available snippet field
                snippet = None
                for field in ("text", "snippet", "content", "transcript", "body"):
                    if field in meta:
                        snippet = meta[field]
                        break
                if snippet is None:
                    snippet = "<no snippet available>"
                
                # Pick a filename (or fallback to 'source')
                filename = meta.get("filename", meta.get("source", "unknown"))
                
                # Format the result
                formatted_match = f"[{i+1}] [{filename}]\n{snippet}\n\n"
                all_results.append({
                    "filename": filename,
                    "content": snippet,
                    "score": match.score
                })
                
                formatted_results += formatted_match
                
                # Stream this match if callback provided
                if stream_callback:
                    await stream_callback(f"\n[{i+1}] [{filename}]\n")
                    
                    # Stream the snippet in chunks for more natural flow
                    chunk_size = 40
                    snippet_chunks = [snippet[i:i+chunk_size] for i in range(0, len(snippet), chunk_size)]
                    for chunk in snippet_chunks:
                        await stream_callback(chunk)
                        await asyncio.sleep(0.05)
                    
                    await stream_callback("\n\n")
            
            # Return the full formatted results
            return formatted_results
            
    except Exception as e:
        error_message = f"Exception in interview search: {str(e)}"
        print(error_message)
        
        # If there's an exception when streaming, send a simplified error message
        if stream_callback and callable(stream_callback):
            await stream_callback(f"Error occurred during interview search: {str(e)[:50]}")
            
        return error_message

async def fallback_search(query, stream_callback=None):
    """Perform a fallback search using OpenAI when Perplexity fails."""
    print(f"Using fallback search for query: '{query}'")
    
    if not OPENAI_API_KEY:
        error_message = "OpenAI API key not available for fallback search."
        print(error_message)
        if stream_callback and callable(stream_callback):
            await stream_callback("Error: Unable to perform fallback search.")
        return error_message
    
    try:
        # Craft a prompt that asks for a factual response about the query
        prompt = f"""Please provide a factual, up-to-date summary about: {query}
        
Focus on providing accurate information only. Include key facts and figures if relevant.
If the information would require very recent data that might not be in your training data,
please mention that limitation.
        
Respond in a concise, informative manner without any filler text or disclaimers."""
        
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        
        body = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant providing factual information. Always address the user as KARTIK in your responses."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,  # Keep temperature low for more factual responses
            "max_tokens": 800
        }
        
        if stream_callback:
            await stream_callback("Using AI to generate information about your query...")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            print("Making OpenAI API request for fallback search")
            response = await client.post(
                OPENAI_URL,
                headers=headers,
                json=body
            )
            
            print(f"OpenAI API response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                result = data["choices"][0]["message"]["content"]
                print(f"Fallback search result (truncated): {result[:100]}...")
                
                # Send each paragraph separately for more natural streaming
                if stream_callback and callable(stream_callback):
                    paragraphs = result.split("\n\n")
                    for paragraph in paragraphs:
                        if paragraph.strip():
                            # Further chunk the paragraph for smoother streaming
                            chunk_size = 40
                            for i in range(0, len(paragraph), chunk_size):
                                chunk = paragraph[i:i+chunk_size]
                                await stream_callback(chunk)
                                await asyncio.sleep(0.05)
                            
                            # Add a newline between paragraphs
                            await stream_callback("\n\n")
                
                # Add a note about where the information came from
                note = "\n\nNote: This information was generated using AI and may not include the very latest developments."
                if stream_callback:
                    await stream_callback(note)
                
                return result + note
            else:
                error_message = f"Error in fallback search: Status {response.status_code}"
                print(error_message)
                if stream_callback:
                    await stream_callback("Unable to retrieve information from fallback search.")
                return error_message
    except Exception as e:
        error_message = f"Exception in fallback search: {str(e)}"
        print(error_message)
        if stream_callback:
            await stream_callback("An error occurred during the fallback search.")
        return error_message

@app.post("/api/my-custom-chat")
async def custom_chat(request: Request):
    payload = await request.json()
    system_prompt = payload.get("system", "")
    tools = payload.get("tools", [])
    user_messages = payload.get("messages", [])
    
    # Add default tools if not present
    if not tools:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for real-time information about any topic.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to look up on the web."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "interview_search",
                    "description": "Search through interview transcripts for relevant information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to find relevant interview transcript segments."
                            },
                            "company_name": {
                                "type": "string",
                                "description": "The company namespace to search within. Defaults to 'innabox'."
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
    
    # Convert message format from content array to simple text for OpenAI
    openai_messages = []
    
    # Get our enhanced system prompt with table data
    enhanced_system_prompt = await get_system_prompt_with_table_data()
    
    # Add system message - use enhanced prompt or user-provided one
    if system_prompt:
        # Combine our enhanced prompt with user-provided one
        openai_messages.append({"role": "system", "content": f"{enhanced_system_prompt}\n\n{system_prompt}"})
    else:
        openai_messages.append({"role": "system", "content": enhanced_system_prompt})
    
    # Convert user messages to OpenAI format
    for msg in user_messages:
        role = msg.get("role", "user")
        content_parts = msg.get("content", [])
        
        # Extract text parts and join them
        text_content = ""
        for part in content_parts:
            if part.get("type") == "text":
                text_content += part.get("text", "")
        
        openai_messages.append({"role": role, "content": text_content})

    async def custom_stream():
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        
        body = {
            "model": "gpt-4o",
            "messages": openai_messages,  # We've already added the system message above
            "stream": True,
            # "max_tokens": 500  # Limit maximum response length
        }
        
        # Add tools if provided
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        # Create a reference to the yield function for callbacks
        output_queue = asyncio.Queue()
        
        async def yield_to_queue(content):
            await output_queue.put(content)
        
        # Process task
        async def process_api():
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    OPENAI_URL,
                    headers=headers,
                    json=body
                ) as resp:
                    if resp.status_code != 200:
                        try:
                            # For streaming responses, we shouldn't access .text() directly
                            # Instead, read the response content manually
                            content = b""
                            async for chunk in resp.aiter_bytes():
                                content += chunk
                            error_text = content.decode('utf-8', errors='replace')
                        except Exception as e:
                            error_text = f"Error reading response (status {resp.status_code}): {str(e)}"
                        
                        error_text = error_text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
                        await yield_to_queue(f"0:\"{error_text}\"\n")
                        return
                    
                    # Variables to track tool calls
                    current_tool_calls = []
                    
                    async for line in resp.aiter_lines():
                        if line and line.startswith("data: "):
                            line = line[6:]  # Remove "data: " prefix
                            
                            # Skip [DONE]
                            if line == "[DONE]":
                                continue
                            
                            try:
                                # Parse JSON response
                                data = json.loads(line)
                                
                                # Extract content delta if available
                                if "choices" in data and len(data["choices"]) > 0:
                                    choice = data["choices"][0]
                                    index = choice.get("index", 0)
                                    
                                    # Handle tool calls
                                    if "delta" in choice and "tool_calls" in choice["delta"]:
                                        tool_call_delta = choice["delta"]["tool_calls"]
                                        
                                        # Initialize tool calls list if this is the first delta
                                        if len(current_tool_calls) == 0 and len(tool_call_delta) > 0:
                                            for _ in range(len(tool_call_delta)):
                                                current_tool_calls.append({
                                                    "id": None,
                                                    "type": "function",
                                                    "function": {"name": "", "arguments": ""}
                                                })
                                        
                                        # Update tool calls with delta information
                                        for i, call_delta in enumerate(tool_call_delta):
                                            if i < len(current_tool_calls):
                                                # Update tool call ID
                                                if "id" in call_delta:
                                                    current_tool_calls[i]["id"] = call_delta["id"]
                                                
                                                # Update function information
                                                if "function" in call_delta:
                                                    if "name" in call_delta["function"]:
                                                        current_tool_calls[i]["function"]["name"] = call_delta["function"]["name"]
                                                    if "arguments" in call_delta["function"]:
                                                        current_tool_calls[i]["function"]["arguments"] += call_delta["function"]["arguments"]
                                    
                                    # Execute tool calls when complete
                                    if choice.get("finish_reason") == "tool_calls" and len(current_tool_calls) > 0:
                                        # Create a new messages array with the previous messages
                                        new_messages = openai_messages.copy()
                                        
                                        # Add the assistant's message with the tool calls
                                        new_messages.append({
                                            "role": "assistant",
                                            "tool_calls": current_tool_calls
                                        })
                                        
                                        # Process each tool call
                                        for tool_call in current_tool_calls:
                                            function_name = tool_call["function"]["name"]
                                            
                                            if function_name == "web_search":
                                                try:
                                                    arguments = json.loads(tool_call["function"]["arguments"])
                                                    query = arguments.get("query", "")
                                                    
                                                    # Send initial search message - special format tag for tool start
                                                    await yield_to_queue(f"{index}:\"<<TOOL_START>>\"\n")
                                                    
                                                    # Now send the actual search message with a special prefix
                                                    search_message = f"Searching for {query}..."
                                                    search_message_escaped = search_message.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
                                                    await yield_to_queue(f"{index}:\"{search_message_escaped}\"\n") # Added prefix
                                                    await asyncio.sleep(0.1) # Ensure it displays
                                                    
                                                    # Execute web search with streaming
                                                    async def stream_callback(chunk):
                                                        print(f"Web search chunk: {chunk[:50]}..." if len(chunk) > 50 else f"Web search chunk: {chunk}")
                                                        escaped_chunk = chunk.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
                                                        await yield_to_queue(f"{index}:\"{escaped_chunk}\"\n")
                                                        await asyncio.sleep(0.05)
                                                    
                                                    search_result = await web_search(query, stream_callback=stream_callback)
                                                    print(f"Web search complete. Result length: {len(search_result)}")
                                                    
                                                    # Check if the result is an error message or empty
                                                    if search_result.startswith("Error") or "error" in search_result.lower() or len(search_result.strip()) < 20:
                                                        print(f"Web search failed or returned minimal results: {search_result}")
                                                        
                                                        # Provide a useful fallback message to the user
                                                        fallback_message = f"I couldn't retrieve current search results for '{query}'. This might be due to API limitations. Please try again later or rephrase your query."
                                                        
                                                        # Stream the fallback message
                                                        await stream_callback(fallback_message)
                                                        
                                                        # Use the fallback message as the search result
                                                        search_result = fallback_message
                                                    
                                                    # Add the function result to messages
                                                    new_messages.append({
                                                        "role": "tool",
                                                        "tool_call_id": tool_call["id"],
                                                        "content": search_result
                                                    })
                                                    
                                                    # Send the tool end marker
                                                    await yield_to_queue(f"{index}:\"<<TOOL_END>>\"\n")
                                                except Exception as e:
                                                    # Handle errors in function execution
                                                    error_message = f"Error executing web_search: {str(e)}"
                                                    error_message = error_message.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
                                                    new_messages.append({
                                                        "role": "tool",
                                                        "tool_call_id": tool_call["id"],
                                                        "content": error_message
                                                    })
                                                    
                                                    # Send error with tool start/end tags
                                                    await yield_to_queue(f"{index}:\"<<TOOL_START>>\"\n")
                                                    error_message = error_message.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
                                                    await yield_to_queue(f"{index}:\"{error_message}\"\n")
                                                    await yield_to_queue(f"{index}:\"<<TOOL_END>>\"\n")
                                            elif function_name == "interview_search":
                                                try:
                                                    arguments = json.loads(tool_call["function"]["arguments"])
                                                    query = arguments.get("query", "")
                                                    company_name = arguments.get("company_name", "innabox")
                                                    
                                                    # Send initial search message
                                                    await yield_to_queue(f"{index}:\"<<TOOL_START>>\"\n")
                                                    
                                                    search_message = f"Searching interview transcripts for '{query}' in company '{company_name}'..."
                                                    search_message = search_message.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
                                                    await yield_to_queue(f"{index}:\"{search_message}\"\n")
                                                    
                                                    # Execute interview search with streaming
                                                    async def stream_callback(chunk):
                                                        escaped_chunk = chunk.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
                                                        await yield_to_queue(f"{index}:\"{escaped_chunk}\"\n")
                                                    
                                                    search_result = await interview_search(query, company_name, stream_callback=stream_callback)
                                                    
                                                    # Add the function result to messages
                                                    new_messages.append({
                                                        "role": "tool",
                                                        "tool_call_id": tool_call["id"],
                                                        "content": search_result
                                                    })
                                                    
                                                    # Send the tool end marker
                                                    await yield_to_queue(f"{index}:\"<<TOOL_END>>\"\n")
                                                except Exception as e:
                                                    # Handle errors in function execution
                                                    error_message = f"Error executing interview_search: {str(e)}"
                                                    error_message = error_message.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
                                                    new_messages.append({
                                                        "role": "tool",
                                                        "tool_call_id": tool_call["id"],
                                                        "content": error_message
                                                    })
                                                    
                                                    # Send error with tool start/end tags
                                                    await yield_to_queue(f"{index}:\"<<TOOL_START>>\"\n")
                                                    error_message = error_message.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
                                                    await yield_to_queue(f"{index}:\"{error_message}\"\n")
                                                    await yield_to_queue(f"{index}:\"<<TOOL_END>>\"\n")
                                        
                                        # Call OpenAI again with the results of the function call
                                        second_response = await client.post(
                                            OPENAI_URL,
                                            headers=headers,
                                            json={
                                                "model": "gpt-4o",
                                                "messages": new_messages,  # We now have the system message in new_messages already
                                                "stream": True,
                                                "temperature": 0.7,  # Add temperature to ensure valid responses
                                                "max_tokens": 1000  # Increase max_tokens to ensure complete responses
                                            }
                                        )
                                        
                                        # Add a delay before starting the second response streaming
                                        # This gives the UI time to process the search results
                                        await asyncio.sleep(0.5)
                                        
                                        # Add a separator to indicate the transition from search to AI response
                                        separator = "\n\nBased on the search results, here's my response:"
                                        separator = separator.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
                                        await yield_to_queue(f"{index}:\"{separator}\"\n")
                                        await asyncio.sleep(0.5)
                                        
                                        if second_response.status_code == 200:
                                            # Stream the second response
                                            finished_streaming = False
                                            async for second_line in second_response.aiter_lines():
                                                if second_line and second_line.startswith("data: "):
                                                    second_line = second_line[6:]
                                                    
                                                    if second_line == "[DONE]":
                                                        # Mark that we've finished streaming the response
                                                        finished_streaming = True
                                                        break  # Exit the streaming loop
                                                    
                                                    try:
                                                        second_data = json.loads(second_line)
                                                        
                                                        if "choices" in second_data and len(second_data["choices"]) > 0:
                                                            second_choice = second_data["choices"][0]
                                                            second_index = second_choice.get("index", 0)
                                                            
                                                            # Get content from delta
                                                            if "delta" in second_choice and "content" in second_choice["delta"]:
                                                                content = second_choice["delta"]["content"]
                                                                if content:
                                                                    # Format as index:"content" with newline escaping
                                                                    escaped_content = content.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
                                                                    await yield_to_queue(f"{second_index}:\"{escaped_content}\"\n")
                                                                    # Add a small delay between tokens for smoother streaming
                                                                    await asyncio.sleep(0.02)
                                                    except json.JSONDecodeError:
                                                        # Skip invalid JSON
                                                        continue
                                        else:
                                            try:
                                                # For streaming responses, we shouldn't access .text() directly
                                                # Instead, read the response content manually
                                                content = b""
                                                async for chunk in second_response.aiter_bytes():
                                                    content += chunk
                                                error_text = content.decode('utf-8', errors='replace')
                                            except Exception as e:
                                                error_text = f"Error reading response (status {second_response.status_code}): {str(e)}"
                                            
                                            error_text = error_text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
                                            await yield_to_queue(f"0:\"{error_text}\"\n")
                                    
                                    # Get content from delta (for normal responses)
                                    elif "delta" in choice and "content" in choice["delta"]:
                                        content = choice["delta"]["content"]
                                        if content:
                                            # Format as index:"content" with newline escaping
                                            escaped_content = content.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
                                            await yield_to_queue(f"{index}:\"{escaped_content}\"\n")
                                            # Add a small delay between tokens for smoother streaming
                                            await asyncio.sleep(0.02)
                                    
                                    # If finish_reason is stop, send [DONE] to terminate stream
                                    if choice.get("finish_reason") == "stop":
                                        break
                                    
                            except json.JSONDecodeError:
                                # Skip invalid JSON
                                continue
        
        # Start the processing task
        task = asyncio.create_task(process_api())
        
        # Yield from the queue as items become available
        try:
            last_message_time = asyncio.get_event_loop().time()
            
            while True:
                try:
                    item = await asyncio.wait_for(output_queue.get(), timeout=1.0)  # Reduce timeout for faster detection of completion
                    yield item
                    output_queue.task_done()
                    
                    # Update the last message time
                    last_message_time = asyncio.get_event_loop().time()
                    
                    # If this is the [DONE] marker, break the loop to close the connection
                    if item.strip() == "0:\"done\"":
                        # Give a little time for client to process the [DONE] marker
                        await asyncio.sleep(0.1)
                        break
                except asyncio.TimeoutError:
                    # Check if we've been idle too long (15 seconds)
                    current_time = asyncio.get_event_loop().time()
                    if current_time - last_message_time > 50:
                        # Force completion after 50 seconds of inactivity
                        break
                        
                    # Check if the task is done
                    if task.done():
                        break
        finally:
            # Make sure to clean up the task
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
        # No need for an additional [DONE] at the end, as we've already sent it

    return StreamingResponse(
        custom_stream(),
        media_type="text/plain",
        headers={"Connection": "close"}  # Explicitly tell client to close the connection
    )

@app.post("/api/chat")
async def chat(request: Request):
    """Alias for /api/my-custom-chat to make it compatible with frontend"""
    return await custom_chat(request)

@app.get("/api/test-perplexity")
async def test_perplexity():
    """Test endpoint to verify Perplexity API connectivity."""
    if not PERPLEXITY_API_KEY:
        return {"status": "error", "message": "PERPLEXITY_API_KEY not set"}
    
    try:
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json",
        }
        
        body = {
            "model": "sonar-reasoning-pro",
            "messages": [{"role": "user", "content": "What day is it today?"}],
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                PERPLEXITY_URL,
                headers=headers,
                json=body
            )
            
            if response.status_code == 200:
                data = response.json()
                result = data.get("choices", [{}])[0].get("message", {}).get("content", "No content")
                return {
                    "status": "success",
                    "message": "Perplexity API is working",
                    "response_code": response.status_code,
                    "result_sample": result[:100] + "..." if len(result) > 100 else result
                }
            else:
                return {
                    "status": "error",
                    "message": f"Perplexity API returned status code {response.status_code}",
                    "response": response.text
                }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Exception while testing Perplexity API: {str(e)}"
        }

@app.post("/api/update-perplexity-key")
async def update_perplexity_key(request: Request):
    """Update the Perplexity API key at runtime."""
    global PERPLEXITY_API_KEY
    
    try:
        data = await request.json()
        new_key = data.get("api_key")
        
        if not new_key:
            return {"status": "error", "message": "No API key provided"}
        
        # Store the old key to revert if testing fails
        old_key = PERPLEXITY_API_KEY
        
        # Update the key
        PERPLEXITY_API_KEY = new_key
        
        # Test the new key
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json",
        }
        
        body = {
            "model": "sonar-reasoning-pro",
            "messages": [{"role": "user", "content": "Test message"}],
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                PERPLEXITY_URL,
                headers=headers,
                json=body
            )
            
            if response.status_code == 200:
                return {
                    "status": "success",
                    "message": "Perplexity API key updated and verified",
                    "key_preview": f"{PERPLEXITY_API_KEY[:8]}..."
                }
            else:
                # Revert to the old key if the new one doesn't work
                PERPLEXITY_API_KEY = old_key
                return {
                    "status": "error",
                    "message": f"New API key validation failed with status code {response.status_code}",
                    "response": response.text
                }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Exception while updating Perplexity API key: {str(e)}"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=3000, reload=True)