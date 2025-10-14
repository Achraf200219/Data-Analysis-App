"""
Enhanced OpenRouter API client for handling multiple data source types.

This module provides OpenRouter API integration for both database files (SQL generation)
and tabular files (pandas code generation), with session management for multiple files.
"""

import os
import json
import sqlite3
import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from urllib.parse import urlparse

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objects import Figure

from file_handler import DataSource, FileType

logger = logging.getLogger(__name__)


class EnhancedOpenRouterClient:
    """Enhanced OpenRouter API client that handles multiple data source types."""

    def __init__(self, api_key: str, model: str = "openai/gpt-4o", base_url: str = "https://openrouter.ai/api/v1"):
        """Initialize the OpenRouter client.
        
        Args:
            api_key: OpenRouter API key
            model: Model to use (default: openai/gpt-4o)
            base_url: OpenRouter base URL
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/data-analysis-app",
            "X-Title": "Data Analysis App"
        }
        self.client = httpx.AsyncClient(headers=self.headers, timeout=60.0)
        
        # Session management for multiple data sources
        self.data_sources: Dict[str, DataSource] = {}
        self.active_source: Optional[str] = None
        
        # Rate limiting configuration
        self.max_retries = 3
        self.base_delay = 1.0  # Base delay in seconds
        self.max_delay = 60.0  # Maximum delay in seconds
        self.last_request_time = 0
        self.min_request_interval = 0.1  # Minimum time between requests

    async def _make_request(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """Make a request to OpenRouter API with rate limiting and retry logic."""
        
        for attempt in range(self.max_retries + 1):
            try:
                # Rate limiting: ensure minimum interval between requests
                current_time = time.time()
                time_since_last_request = current_time - self.last_request_time
                if time_since_last_request < self.min_request_interval:
                    await asyncio.sleep(self.min_request_interval - time_since_last_request)
                
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                
                self.last_request_time = time.time()
                response = await self.client.post(f"{self.base_url}/chat/completions", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"].strip()
                
                elif response.status_code == 429:  # Rate limit exceeded
                    if attempt < self.max_retries:
                        # Calculate exponential backoff delay
                        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                        
                        # Check for Retry-After header
                        retry_after = response.headers.get('retry-after')
                        if retry_after:
                            try:
                                delay = float(retry_after)
                            except ValueError:
                                pass
                        
                        logger.warning(f"Rate limit hit (429), retrying in {delay} seconds (attempt {attempt + 1}/{self.max_retries})")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.error(f"Rate limit exceeded after {self.max_retries} retries")
                        raise Exception("Rate limit exceeded. Please try again later.")
                
                elif response.status_code >= 500:  # Server errors
                    if attempt < self.max_retries:
                        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                        logger.warning(f"Server error ({response.status_code}), retrying in {delay} seconds (attempt {attempt + 1}/{self.max_retries})")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.error(f"Server error after {self.max_retries} retries: {response.status_code}")
                        raise Exception(f"Server error: {response.status_code}")
                
                else:
                    # Other HTTP errors (4xx client errors)
                    response.raise_for_status()
                    
            except httpx.HTTPStatusError as e:
                if attempt < self.max_retries and e.response.status_code >= 500:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    logger.warning(f"HTTP error ({e.response.status_code}), retrying in {delay} seconds (attempt {attempt + 1}/{self.max_retries})")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"OpenRouter API error: {e.response.status_code} - {e.response.text}")
                    raise Exception(f"OpenRouter API error: {e.response.status_code}")
            except httpx.TimeoutException:
                if attempt < self.max_retries:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    logger.warning(f"Request timeout, retrying in {delay} seconds (attempt {attempt + 1}/{self.max_retries})")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"Request timeout after {self.max_retries} retries")
                    raise Exception("Request timeout. Please try again later.")
            except Exception as e:
                logger.error(f"Request failed: {str(e)}")
                if attempt < self.max_retries:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    logger.warning(f"Unexpected error, retrying in {delay} seconds (attempt {attempt + 1}/{self.max_retries})")
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise
        
        # If we get here, all retries failed
        raise Exception("All retry attempts failed")

    async def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status from OpenRouter."""
        try:
            response = await self.client.get(f"{self.base_url}/auth/key")
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "active",
                    "rate_limit": data.get("rate_limit", {}),
                    "usage": data.get("usage", {}),
                    "limits": data.get("limits", {})
                }
            else:
                return {"status": "error", "code": response.status_code}
        except Exception as e:
            logger.error(f"Failed to get rate limit status: {e}")
            return {"status": "unknown", "error": str(e)}

    # --- Data Source Management ---
    
    def add_data_source(self, source_id: str, data_source: DataSource):
        """Add a data source to the session"""
        self.data_sources[source_id] = data_source
        if self.active_source is None:
            self.active_source = source_id
    
    def set_active_source(self, source_id: str):
        """Set the active data source"""
        if source_id not in self.data_sources:
            raise ValueError(f"Data source {source_id} not found")
        self.active_source = source_id
    
    def get_active_source(self) -> Optional[DataSource]:
        """Get the current active data source"""
        if self.active_source is None:
            return None
        return self.data_sources.get(self.active_source)
    
    def get_data_sources(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all loaded data sources"""
        sources_info = {}
        for source_id, data_source in self.data_sources.items():
            sources_info[source_id] = {
                "name": data_source.name,
                "type": data_source.file_type.value,
                "metadata": data_source.metadata,
                "schema": data_source.get_schema()
            }
        return sources_info
    
    def remove_data_source(self, source_id: str):
        """Remove a data source from the session"""
        if source_id in self.data_sources:
            self.data_sources[source_id].close()
            del self.data_sources[source_id]
            if self.active_source == source_id:
                self.active_source = next(iter(self.data_sources.keys())) if self.data_sources else None
    
    def _get_data_source(self, source_id: str = None) -> Optional[DataSource]:
        """Get data source by ID or return active source"""
        if source_id is not None:
            return self.data_sources.get(source_id)
        return self.get_active_source()
    
    def _is_database_source(self, data_source: DataSource) -> bool:
        """Check if data source is a database type"""
        return data_source.file_type in [FileType.SQLITE, FileType.DUCKDB, FileType.ACCESS, FileType.SQL_DUMP]
    
    def _format_schema_for_prompt(self, data_source: DataSource, schema: Dict[str, Any]) -> str:
        """Format schema for AI prompts based on data source type"""
        if self._is_database_source(data_source):
            return self._format_database_schema(schema)
        else:
            return self._format_dataframe_schema(schema)
    
    def _format_database_schema(self, schema: Dict[str, Any]) -> str:
        """Format database schema for prompts"""
        if "error" in schema:
            return f"Schema error: {schema['error']}"
        
        schema_parts = []
        for table_info in schema.get("tables", []):
            table_name = table_info["name"]
            columns = ", ".join([f"{col['name']} ({col['type']})" for col in table_info["columns"]])
            schema_parts.append(f"Table: {table_name}\nColumns: {columns}")
            
            if table_info.get("sample_data"):
                sample = str(table_info["sample_data"][:2])  # Show 2 sample rows
                schema_parts.append(f"Sample data: {sample}")
        
        return "\n\n".join(schema_parts)
    
    def _format_dataframe_schema(self, schema: Dict[str, Any]) -> str:
        """Format dataframe schema for prompts"""
        if "error" in schema:
            return f"Schema error: {schema['error']}"
        
        schema_parts = []
        schema_parts.append(f"Dataset with {schema.get('total_rows', 0)} rows and {schema.get('total_columns', 0)} columns")
        
        schema_parts.append("Columns:")
        for col_info in schema.get("columns", []):
            col_desc = f"- {col_info['name']} ({col_info['type']})"
            if col_info.get('sample_values'):
                col_desc += f" - Sample values: {col_info['sample_values'][:3]}"
            schema_parts.append(col_desc)
        
        if schema.get("sample_data"):
            sample = str(schema["sample_data"][:2])  # Show 2 sample rows
            schema_parts.append(f"\nSample data: {sample}")
        
        return "\n".join(schema_parts)

    # --- Query Generation ---
    
    async def generate_questions(self, source_id: str = None) -> List[str]:
        """Generate sample questions based on data source schema."""
        data_source = self._get_data_source(source_id)
        
        if data_source is None:
            return [
                "What are the main trends in this data?",
                "Show me the top 10 records",
                "What is the distribution of values in the main columns?",
                "Can you summarize the key statistics?"
            ]

        schema = data_source.get_schema()
        schema_description = self._format_schema_for_prompt(data_source, schema)
        
        messages = [
            {
                "role": "system",
                "content": f"You are a helpful assistant that generates sample questions for data exploration. Generate 5-8 interesting business questions that can be answered with {'SQL queries' if self._is_database_source(data_source) else 'pandas operations'} based on the provided data schema."
            },
            {
                "role": "user",
                "content": f"Based on this {'database' if self._is_database_source(data_source) else 'dataset'} schema, generate sample questions:\n\n{schema_description}\n\nReturn only the questions, one per line, without numbering or bullets."
            }
        ]
        
        response = await self._make_request(messages, temperature=0.8)
        questions = [q.strip() for q in response.split('\n') if q.strip()]
        return questions[:8]  # Limit to 8 questions

    async def generate_query(self, question: str, source_id: str = None) -> Dict[str, Any]:
        """Generate appropriate query (SQL or pandas) based on data source type."""
        data_source = self._get_data_source(source_id)
        
        if data_source is None:
            raise ValueError("No active data source")
        
        schema = data_source.get_schema()
        schema_description = self._format_schema_for_prompt(data_source, schema)
        
        if self._is_database_source(data_source):
            return await self._generate_sql_query(question, schema_description, data_source)
        else:
            return await self._generate_pandas_query(question, schema_description, data_source)
    
    async def _generate_sql_query(self, question: str, schema_description: str, data_source: DataSource) -> Dict[str, Any]:
        """Generate SQL query for database sources."""
        db_type = "SQLite" if data_source.file_type == FileType.SQLITE else data_source.file_type.value.upper()
        
        messages = [
            {
                "role": "system",
                "content": f"""You are an expert SQL query generator. Generate accurate, efficient SQL queries based on natural language questions and the provided database schema. 
                
Rules:
- Return ONLY the SQL query, no explanations or markdown
- Use {db_type} syntax
- Include appropriate JOINs when needed
- Add sensible LIMIT clauses for large result sets
- Use proper table and column names from the schema
- Handle case sensitivity appropriately"""
            },
            {
                "role": "user",
                "content": f"Database schema:\n{schema_description}\n\nQuestion: {question}\n\nGenerate a SQL query to answer this question:"
            }
        ]
        
        sql = await self._make_request(messages, temperature=0.1)
        
        # Clean up the SQL - remove any markdown formatting
        sql = sql.replace("```sql", "").replace("```", "").strip()
        
        return {
            "query_type": "sql",
            "query": sql,
            "data_source": data_source.name,
            "source_type": data_source.file_type.value
        }
    
    async def _generate_pandas_query(self, question: str, schema_description: str, data_source: DataSource) -> Dict[str, Any]:
        """Generate pandas code for tabular data sources."""
        messages = [
            {
                "role": "system",
                "content": """You are an expert pandas data analyst. Generate efficient pandas code based on natural language questions and the provided dataset schema.

Rules:
- Use 'df' as the DataFrame variable name
- Return ONLY the Python pandas code, no explanations or markdown
- Use appropriate pandas methods for filtering, grouping, aggregating
- Store the final result in a variable called 'result'
- Handle data types appropriately
- Include proper column names from the schema
- For display purposes, limit results to reasonable sizes (e.g., .head(20) for large results)"""
            },
            {
                "role": "user",
                "content": f"Dataset schema:\n{schema_description}\n\nQuestion: {question}\n\nGenerate pandas code to answer this question:"
            }
        ]
        
        code = await self._make_request(messages, temperature=0.1)
        
        # Clean up the code
        code = code.replace("```python", "").replace("```", "").strip()
        
        return {
            "query_type": "pandas",
            "query": code,
            "data_source": data_source.name,
            "source_type": data_source.file_type.value
        }

    # --- Query Execution ---
    
    def execute_query(self, query_info: Dict[str, Any], source_id: str = None) -> pd.DataFrame:
        """Execute a query (SQL or pandas) and return results."""
        data_source = self._get_data_source(source_id or query_info.get("data_source"))
        
        if data_source is None:
            raise ValueError("No data source available for query execution")
        
        if query_info["query_type"] == "sql":
            return data_source.execute_sql(query_info["query"])
        elif query_info["query_type"] == "pandas":
            return data_source.execute_pandas(query_info["query"])
        else:
            raise ValueError(f"Unknown query type: {query_info['query_type']}")

    # --- Visualization and Analysis ---
    
    async def should_generate_chart(self, df: pd.DataFrame, source_id: str = None) -> bool:
        """Determine if a chart should be generated for the given DataFrame."""
        if df.empty or len(df) < 2:
            return False
        
        # Simple heuristics for chart worthiness
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            return False
        
        # If we have few rows but numeric data, it might be good for charting
        if len(df) <= 50 and len(numeric_cols) >= 1:
            return True
        
        return len(df) <= 1000  # Don't chart very large datasets

    async def generate_plotly_code(self, question: str, query_info: Dict[str, Any], df: pd.DataFrame, source_id: str = None) -> str:
        """Generate Plotly code for visualizing the data."""
        if df.empty:
            return ""
        
        # Prepare data description
        data_description = self._describe_dataframe(df)
        query_text = query_info.get("query", "")
        
        messages = [
            {
                "role": "system",
                "content": """You are an expert in data visualization with Plotly. Generate Python code using plotly.express or plotly.graph_objects to create appropriate charts.

Rules:
- Return ONLY the Python code, no explanations
- Use 'df' as the DataFrame variable name
- Import statements should be: import plotly.express as px, import plotly.graph_objects as go
- Choose appropriate chart types based on the data
- Include proper titles and axis labels
- The code should create a variable called 'fig'
- Handle categorical vs numerical data appropriately"""
            },
            {
                "role": "user",
                "content": f"Question: {question}\n{query_info['query_type'].upper()}: {query_text}\nData description: {data_description}\n\nGenerate Plotly code to visualize this data:"
            }
        ]
        
        code = await self._make_request(messages, temperature=0.3)
        
        # Clean up the code
        code = code.replace("```python", "").replace("```", "").strip()
        
        return code

    def get_plotly_figure(self, plotly_code: str, df: pd.DataFrame, source_id: str = None) -> Figure:
        """Execute Plotly code and return the figure."""
        try:
            # Create execution environment
            exec_env = {
                'df': df,
                'pd': pd,
                'px': px,
                'go': go,
                'fig': None
            }
            
            # Execute the code
            exec(plotly_code, exec_env)
            
            fig = exec_env.get('fig')
            if fig is None:
                raise Exception("Generated code did not create a 'fig' variable")
            
            return fig
        except Exception as e:
            logger.error(f"Failed to execute Plotly code: {str(e)}")
            # Return a simple fallback chart
            return px.bar(x=['Error'], y=[1], title="Chart generation failed")

    async def generate_summary(self, question: str, df: pd.DataFrame, query_info: Dict[str, Any] = None, source_id: str = None) -> Optional[str]:
        """Generate a textual summary of the query results."""
        if df.empty:
            return "No data returned from the query."
        
        # Prepare data summary
        data_summary = self._create_data_summary(df)
        query_text = query_info.get("query", "") if query_info else ""
        
        messages = [
            {
                "role": "system",
                "content": "You are a data analyst. Provide a concise, insightful summary of query results. Focus on key findings, trends, and notable patterns. Keep it under 150 words."
            },
            {
                "role": "user",
                "content": f"Question: {question}\nQuery: {query_text}\nData summary: {data_summary}\n\nProvide an insightful summary of these results:"
            }
        ]
        
        try:
            summary = await self._make_request(messages, temperature=0.5, max_tokens=200)
            return summary
        except Exception as e:
            logger.error(f"Failed to generate summary: {str(e)}")
            return f"Query returned {len(df)} rows with {len(df.columns)} columns."

    async def generate_followup_questions(self, question: str, query_info: Dict[str, Any], df: pd.DataFrame, source_id: str = None) -> List[str]:
        """Generate follow-up questions based on the results."""
        if df.empty:
            return []
        
        data_summary = self._create_data_summary(df)
        query_text = query_info.get("query", "")
        
        messages = [
            {
                "role": "system",
                "content": "Generate 3-5 insightful follow-up questions based on the original question and data results. Questions should be specific, actionable, and explore different angles or deeper insights."
            },
            {
                "role": "user",
                "content": f"Original question: {question}\nQuery: {query_text}\nData summary: {data_summary}\n\nGenerate follow-up questions (one per line, no numbering):"
            }
        ]
        
        try:
            response = await self._make_request(messages, temperature=0.7)
            questions = [q.strip() for q in response.split('\n') if q.strip()]
            return questions[:5]  # Limit to 5 questions
        except Exception as e:
            logger.error(f"Failed to generate follow-up questions: {str(e)}")
            return []

    def _describe_dataframe(self, df: pd.DataFrame) -> str:
        """Create a description of the DataFrame structure."""
        description = f"DataFrame with {len(df)} rows and {len(df.columns)} columns:\n"
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            sample_values = df[col].dropna().unique()[:5]
            description += f"- {col} ({dtype}): {list(sample_values)}\n"
        
        return description

    def _create_data_summary(self, df: pd.DataFrame) -> str:
        """Create a summary of the DataFrame for AI processing."""
        summary = f"Results: {len(df)} rows, {len(df.columns)} columns\n"
        
        # Add column information
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                stats = f"min: {df[col].min()}, max: {df[col].max()}, mean: {df[col].mean():.2f}"
                summary += f"{col}: {stats}\n"
            else:
                unique_count = df[col].nunique()
                top_values = df[col].value_counts().head(3).to_dict()
                summary += f"{col}: {unique_count} unique values, top: {top_values}\n"
        
        # Add first few rows as sample
        summary += f"\nSample rows:\n{df.head(3).to_string()}"
        
        return summary

    async def close(self):
        """Clean up resources."""
        if self.client:
            await self.client.aclose()
        
        # Close all data sources
        for data_source in self.data_sources.values():
            data_source.close()
        
        self.data_sources.clear()
        self.active_source = None

    # --- Legacy Compatibility Methods ---
    
    async def generate_sql(self, question: str, allow_llm_to_see_data: bool = True, source_id: str = None) -> str:
        """Legacy method - generate SQL query from natural language question."""
        result = await self.generate_query(question, source_id)
        return result["query"]
    
    def run_sql(self, sql: str, source_id: str = None) -> pd.DataFrame:
        """Legacy method - execute SQL query and return results as DataFrame."""
        data_source = self._get_data_source(source_id)
        if data_source is None:
            raise ValueError("No active data source")
        return data_source.execute_sql(sql)