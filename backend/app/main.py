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

# Load environment variables
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

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check if API key is set
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("OPENAI_API_KEY environment variable is not set")
    raise ValueError("OPENAI_API_KEY environment variable is not set")
else:
    logger.info("OPENAI_API_KEY is set (length: %d)", len(api_key))

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 