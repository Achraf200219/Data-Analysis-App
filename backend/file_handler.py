"""
File detection and loading system for multiple data formats.

This module handles automatic detection and loading of various file types:
- Database files: SQLite, DuckDB, MS Access, SQL dumps
- Tabular files: CSV, Excel, Parquet

It provides a unified interface for different data sources while maintaining
type-specific functionality for query generation and execution.
"""

import os
import mimetypes
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from enum import Enum
import logging

import pandas as pd
import sqlite3
import duckdb
import sqlalchemy
from sqlalchemy import create_engine, inspect, text
import numpy as np

from safe_executor import safe_execute_pandas, safe_validate_sql, SafetyError

logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    else:
        return obj


class FileType(Enum):
    """Supported file types"""
    SQLITE = "sqlite"
    DUCKDB = "duckdb"
    ACCESS = "access"
    SQL_DUMP = "sql_dump"
    CSV = "csv"
    EXCEL = "excel"
    PARQUET = "parquet"
    UNKNOWN = "unknown"


class DataSource:
    """Represents a loaded data source with metadata"""
    
    def __init__(self, file_path: str, file_type: FileType, name: str):
        self.file_path = file_path
        self.file_type = file_type
        self.name = name
        self.metadata: Dict[str, Any] = {}
        self.connection = None
        self.dataframe: Optional[pd.DataFrame] = None
        self._schema_cache: Optional[Dict[str, Any]] = None
    
    def get_schema(self) -> Dict[str, Any]:
        """Get schema information for the data source"""
        if self._schema_cache is not None:
            return self._schema_cache
        
        if self.file_type in [FileType.SQLITE, FileType.DUCKDB, FileType.ACCESS]:
            self._schema_cache = self._get_database_schema()
        else:
            self._schema_cache = self._get_dataframe_schema()
        
        return self._schema_cache
    
    def _get_database_schema(self) -> Dict[str, Any]:
        """Get schema for database files"""
        try:
            if self.connection is None:
                return {"tables": [], "error": "No database connection"}
            
            inspector = inspect(self.connection)
            tables = inspector.get_table_names()
            
            schema = {"tables": [], "total_tables": len(tables)}
            
            for table_name in tables:
                columns = inspector.get_columns(table_name)
                table_info = {
                    "name": table_name,
                    "columns": [
                        {
                            "name": col["name"],
                            "type": str(col["type"]),
                            "nullable": col.get("nullable", True),
                            "default": col.get("default")
                        }
                        for col in columns
                    ],
                    "column_count": len(columns)
                }
                
                # Get sample data
                try:
                    sample_query = f"SELECT * FROM {table_name} LIMIT 3"
                    with self.connection.connect() as conn:
                        sample_result = conn.execute(text(sample_query))
                        sample_data = [dict(row._mapping) for row in sample_result]
                    table_info["sample_data"] = sample_data
                except Exception as e:
                    table_info["sample_data"] = []
                    logger.warning(f"Could not get sample data for {table_name}: {e}")
                
                schema["tables"].append(table_info)
            
            return schema
            
        except Exception as e:
            logger.error(f"Error getting database schema: {e}")
            return {"tables": [], "error": str(e)}
    
    def _get_dataframe_schema(self) -> Dict[str, Any]:
        """Get schema for tabular files"""
        if self.dataframe is None:
            return {"columns": [], "error": "No dataframe loaded"}
        
        df = self.dataframe
        schema = {
            "columns": [],
            "shape": convert_numpy_types(df.shape),
            "total_rows": convert_numpy_types(len(df)),
            "total_columns": convert_numpy_types(len(df.columns))
        }
        
        for col in df.columns:
            col_info = {
                "name": col,
                "type": str(df[col].dtype),
                "null_count": convert_numpy_types(df[col].isnull().sum()),
                "unique_count": convert_numpy_types(df[col].nunique()),
                "sample_values": convert_numpy_types(df[col].dropna().head(5).tolist())
            }
            schema["columns"].append(col_info)
        
        # Add sample data - convert all numpy types
        sample_data = df.head(3).to_dict("records")
        schema["sample_data"] = convert_numpy_types(sample_data)
        
        return schema
    
    def execute_sql(self, query: str) -> pd.DataFrame:
        """Execute SQL query on database sources"""
        if self.file_type not in [FileType.SQLITE, FileType.DUCKDB, FileType.ACCESS]:
            raise ValueError(f"SQL execution not supported for {self.file_type}")
        
        if self.connection is None:
            raise ValueError("No database connection available")
        
        try:
            # Validate SQL query for safety
            safe_query = safe_validate_sql(query)
            
            # Execute the validated query
            return pd.read_sql_query(safe_query, self.connection)
            
        except SafetyError as e:
            logger.error(f"SQL safety validation failed: {e}")
            raise ValueError(f"Unsafe SQL query: {e}")
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            raise ValueError(f"Query execution failed: {e}")
    
    def execute_pandas(self, code: str) -> pd.DataFrame:
        """Execute pandas code on dataframe sources"""
        if self.file_type not in [FileType.CSV, FileType.EXCEL, FileType.PARQUET]:
            raise ValueError(f"Pandas execution not supported for {self.file_type}")
        
        if self.dataframe is None:
            raise ValueError("No dataframe available")
        
        try:
            # Use safe pandas execution
            return safe_execute_pandas(code, self.dataframe)
            
        except SafetyError as e:
            logger.error(f"Pandas safety validation failed: {e}")
            raise ValueError(f"Unsafe pandas code: {e}")
        except Exception as e:
            logger.error(f"Pandas execution failed: {e}")
            raise ValueError(f"Code execution failed: {e}")
        
        return result
    
    def close(self):
        """Close connections and clean up resources"""
        if self.connection is not None:
            try:
                self.connection.close()
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")
        
        # Clean up temporary files if any
        if hasattr(self, '_temp_file') and self._temp_file:
            try:
                os.unlink(self._temp_file)
            except Exception as e:
                logger.warning(f"Error removing temp file: {e}")


class FileDetector:
    """Detects file types and creates appropriate data sources"""
    
    @staticmethod
    def detect_file_type(file_path: str, original_filename: str = None) -> FileType:
        """Detect file type based on extension and content"""
        path = Path(original_filename or file_path)
        extension = path.suffix.lower()
        
        # Extension-based detection
        extension_map = {
            '.sqlite': FileType.SQLITE,
            '.sqlite3': FileType.SQLITE,
            '.db': FileType.SQLITE,
            '.duckdb': FileType.DUCKDB,
            '.mdb': FileType.ACCESS,
            '.accdb': FileType.ACCESS,
            '.sql': FileType.SQL_DUMP,
            '.csv': FileType.CSV,
            '.xls': FileType.EXCEL,
            '.xlsx': FileType.EXCEL,
            '.parquet': FileType.PARQUET,
        }
        
        if extension in extension_map:
            return extension_map[extension]
        
        # Content-based detection for ambiguous cases
        try:
            # Check if it's a SQLite database by trying to open it
            if FileDetector._is_sqlite_file(file_path):
                return FileType.SQLITE
            
            # Check if it's a DuckDB file
            if FileDetector._is_duckdb_file(file_path):
                return FileType.DUCKDB
            
            # Try to detect CSV by reading first few lines
            if FileDetector._is_csv_file(file_path):
                return FileType.CSV
                
        except Exception as e:
            logger.warning(f"Content-based detection failed: {e}")
        
        return FileType.UNKNOWN
    
    @staticmethod
    def _is_sqlite_file(file_path: str) -> bool:
        """Check if file is a SQLite database"""
        try:
            conn = sqlite3.connect(file_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            conn.close()
            return True
        except Exception:
            return False
    
    @staticmethod
    def _is_duckdb_file(file_path: str) -> bool:
        """Check if file is a DuckDB database"""
        try:
            conn = duckdb.connect(file_path, read_only=True)
            conn.execute("SHOW TABLES;")
            conn.close()
            return True
        except Exception:
            return False
    
    @staticmethod
    def _is_csv_file(file_path: str) -> bool:
        """Check if file is a CSV by attempting to read it"""
        try:
            pd.read_csv(file_path, nrows=1)
            return True
        except Exception:
            return False


class FileLoader:
    """Loads different file types into DataSource objects"""
    
    @staticmethod
    def load_file(file_path: str, file_type: FileType, original_filename: str = None) -> DataSource:
        """Load a file into a DataSource object"""
        name = Path(original_filename or file_path).stem
        
        data_source = DataSource(file_path, file_type, name)
        
        try:
            if file_type == FileType.SQLITE:
                FileLoader._load_sqlite(data_source)
            elif file_type == FileType.DUCKDB:
                FileLoader._load_duckdb(data_source)
            elif file_type == FileType.ACCESS:
                FileLoader._load_access(data_source)
            elif file_type == FileType.SQL_DUMP:
                FileLoader._load_sql_dump(data_source)
            elif file_type == FileType.CSV:
                FileLoader._load_csv(data_source)
            elif file_type == FileType.EXCEL:
                FileLoader._load_excel(data_source)
            elif file_type == FileType.PARQUET:
                FileLoader._load_parquet(data_source)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            logger.info(f"Successfully loaded {file_type.value} file: {name}")
            return data_source
            
        except Exception as e:
            logger.error(f"Failed to load file {name}: {e}")
            raise
    
    @staticmethod
    def _load_sqlite(data_source: DataSource):
        """Load SQLite database"""
        engine = create_engine(f"sqlite:///{data_source.file_path}")
        data_source.connection = engine
        data_source.metadata["engine_type"] = "sqlite"
    
    @staticmethod
    def _load_duckdb(data_source: DataSource):
        """Load DuckDB database"""
        # DuckDB with SQLAlchemy
        engine = create_engine(f"duckdb:///{data_source.file_path}")
        data_source.connection = engine
        data_source.metadata["engine_type"] = "duckdb"
    
    @staticmethod
    def _load_access(data_source: DataSource):
        """Load MS Access database"""
        # Note: This requires ODBC drivers for Access
        try:
            connection_string = (
                f"DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};"
                f"DBQ={data_source.file_path};"
            )
            engine = create_engine(f"mssql+pyodbc:///?odbc_connect={connection_string}")
            data_source.connection = engine
            data_source.metadata["engine_type"] = "access"
        except Exception as e:
            logger.warning(f"Failed to connect to Access database: {e}")
            raise ValueError("Access database support requires ODBC drivers")
    
    @staticmethod
    def _load_sql_dump(data_source: DataSource):
        """Load SQL dump into SQLite"""
        # Create temporary SQLite database from SQL dump
        temp_db = tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False)
        temp_db.close()
        
        # Read SQL dump and execute it
        with open(data_source.file_path, 'r', encoding='utf-8') as f:
            sql_content = f.read()
        
        conn = sqlite3.connect(temp_db.name)
        conn.executescript(sql_content)
        conn.close()
        
        # Create SQLAlchemy engine for the temp database
        engine = create_engine(f"sqlite:///{temp_db.name}")
        data_source.connection = engine
        data_source.metadata["engine_type"] = "sqlite"
        data_source._temp_file = temp_db.name
    
    @staticmethod
    def _load_csv(data_source: DataSource):
        """Load CSV file"""
        # Try different encodings and separators
        encodings = ['utf-8', 'latin1', 'cp1252']
        separators = [',', ';', '\t']
        
        for encoding in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(data_source.file_path, encoding=encoding, sep=sep, nrows=10)
                    if len(df.columns) > 1:  # Successful parsing likely has multiple columns
                        # Load the full file
                        df = pd.read_csv(data_source.file_path, encoding=encoding, sep=sep)
                        data_source.dataframe = df
                        data_source.metadata.update({
                            "encoding": encoding,
                            "separator": sep,
                            "rows": len(df),
                            "columns": len(df.columns)
                        })
                        return
                except Exception:
                    continue
        
        # Fallback to default pandas behavior
        df = pd.read_csv(data_source.file_path)
        data_source.dataframe = df
        data_source.metadata.update({
            "encoding": "default",
            "separator": "default",
            "rows": len(df),
            "columns": len(df.columns)
        })
    
    @staticmethod
    def _load_excel(data_source: DataSource):
        """Load Excel file"""
        # Read all sheets
        xl_file = pd.ExcelFile(data_source.file_path)
        sheets = {}
        
        for sheet_name in xl_file.sheet_names:
            sheets[sheet_name] = pd.read_excel(data_source.file_path, sheet_name=sheet_name)
        
        # Use first sheet as primary dataframe
        if sheets:
            first_sheet = list(sheets.keys())[0]
            data_source.dataframe = sheets[first_sheet]
            data_source.metadata.update({
                "sheets": list(sheets.keys()),
                "active_sheet": first_sheet,
                "all_sheets": sheets,
                "rows": len(data_source.dataframe),
                "columns": len(data_source.dataframe.columns)
            })
    
    @staticmethod
    def _load_parquet(data_source: DataSource):
        """Load Parquet file"""
        df = pd.read_parquet(data_source.file_path)
        data_source.dataframe = df
        data_source.metadata.update({
            "rows": len(df),
            "columns": len(df.columns),
            "memory_usage": df.memory_usage(deep=True).sum()
        })


# Convenience functions for the main application
def detect_and_load_file(file_path: str, original_filename: str = None) -> DataSource:
    """Detect file type and load it into a DataSource"""
    file_type = FileDetector.detect_file_type(file_path, original_filename)
    
    if file_type == FileType.UNKNOWN:
        raise ValueError(f"Unsupported file type: {original_filename or file_path}")
    
    return FileLoader.load_file(file_path, file_type, original_filename)


def get_supported_extensions() -> List[str]:
    """Get list of supported file extensions"""
    return [
        '.sqlite', '.sqlite3', '.db',  # SQLite
        '.duckdb',  # DuckDB
        '.mdb', '.accdb',  # MS Access
        '.sql',  # SQL dumps
        '.csv',  # CSV
        '.xls', '.xlsx',  # Excel
        '.parquet'  # Parquet
    ]