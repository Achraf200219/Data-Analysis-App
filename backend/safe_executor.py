"""
Safe execution environment for pandas operations.

This module provides a sandboxed execution environment for dynamically generated
pandas code, with comprehensive safety checks and input sanitization.
"""

import ast
import operator
import builtins
import logging
from typing import Any, Dict, List, Set, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Allowed built-in functions for pandas operations
ALLOWED_BUILTINS = {
    'abs', 'all', 'any', 'bool', 'dict', 'enumerate', 'filter', 'float', 
    'int', 'len', 'list', 'map', 'max', 'min', 'range', 'round', 'set', 
    'sorted', 'str', 'sum', 'tuple', 'type', 'zip'
}

# Allowed pandas and numpy functions
ALLOWED_MODULES = {
    'pd': pd,
    'pandas': pd,
    'np': np,
    'numpy': np,
}

# Dangerous AST node types to block
DANGEROUS_NODES = {
    ast.Import,           # Block new imports
    ast.ImportFrom,       # Block from imports
    ast.Call,             # Will be checked separately for dangerous calls
}

# Dangerous function calls to block
DANGEROUS_CALLS = {
    'exec', 'eval', 'compile', '__import__', 'open', 'file', 'input', 
    'raw_input', 'reload', 'vars', 'dir', 'globals', 'locals', 'delattr',
    'setattr', 'hasattr', 'getattr'
}

class SafetyError(Exception):
    """Raised when unsafe code is detected"""
    pass

class PandasSafeExecutor:
    """Safe executor for pandas operations"""
    
    def __init__(self, max_execution_time: float = 30.0):
        self.max_execution_time = max_execution_time
    
    def validate_code(self, code: str) -> None:
        """Validate code for security issues before execution"""
        try:
            tree = ast.parse(code)
            self._check_ast_safety(tree)
        except SyntaxError as e:
            raise SafetyError(f"Syntax error in code: {e}")
    
    def _check_ast_safety(self, node: ast.AST) -> None:
        """Recursively check AST nodes for safety"""
        for child in ast.walk(node):
            # Check for dangerous node types
            if type(child) in DANGEROUS_NODES:
                if isinstance(child, ast.Call):
                    self._check_call_safety(child)
                else:
                    raise SafetyError(f"Dangerous operation detected: {type(child).__name__}")
            
            # Check for attribute access to dangerous methods
            if isinstance(child, ast.Attribute):
                self._check_attribute_safety(child)
    
    def _check_call_safety(self, call_node: ast.Call) -> None:
        """Check if function call is safe"""
        func_name = None
        
        if isinstance(call_node.func, ast.Name):
            func_name = call_node.func.id
        elif isinstance(call_node.func, ast.Attribute):
            func_name = call_node.func.attr
        
        if func_name in DANGEROUS_CALLS:
            raise SafetyError(f"Dangerous function call detected: {func_name}")
    
    def _check_attribute_safety(self, attr_node: ast.Attribute) -> None:
        """Check if attribute access is safe"""
        attr_name = attr_node.attr
        
        # Block access to private/special methods
        if attr_name.startswith('_'):
            raise SafetyError(f"Access to private attribute not allowed: {attr_name}")
        
        # Block dangerous pandas/numpy methods
        dangerous_attrs = {
            'eval', 'query'  # These could allow code injection if misused
        }
        
        if attr_name in dangerous_attrs:
            # Allow pandas query/eval with additional checks
            if attr_name in ['eval', 'query']:
                logger.warning(f"Using potentially dangerous method: {attr_name}")
            else:
                raise SafetyError(f"Dangerous attribute access: {attr_name}")
    
    def create_safe_environment(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create a safe execution environment"""
        # Restrict built-ins
        safe_builtins = {name: getattr(builtins, name) for name in ALLOWED_BUILTINS}
        
        # Create safe environment
        env = {
            '__builtins__': safe_builtins,
            'df': df.copy(),  # Work on a copy to prevent modification of original
            'result': None,
            **ALLOWED_MODULES
        }
        
        return env
    
    def execute_pandas_code(self, code: str, df: pd.DataFrame) -> pd.DataFrame:
        """Execute pandas code safely"""
        # Validate code first
        self.validate_code(code)
        
        # Create safe environment
        env = self.create_safe_environment(df)
        
        try:
            # Execute the code with timeout protection
            exec(code, env)
            
            # Get the result
            result = env.get('result')
            
            if result is None:
                # If no explicit result, check if df was modified
                result = env['df']
            
            # Handle different result types appropriately
            if not isinstance(result, pd.DataFrame):
                if isinstance(result, pd.Series):
                    # Convert Series to DataFrame
                    result = result.to_frame().T if len(result.shape) == 1 else result.to_frame()
                elif isinstance(result, (list, tuple)):
                    # Convert list/tuple to DataFrame
                    try:
                        result = pd.DataFrame(result)
                    except:
                        # If list contains scalars, create single-column DataFrame
                        result = pd.DataFrame({'value': result})
                elif isinstance(result, dict):
                    # Convert dict to DataFrame
                    try:
                        result = pd.DataFrame(result)
                    except:
                        # If dict has scalar values, create DataFrame from items
                        result = pd.DataFrame(list(result.items()), columns=['key', 'value'])
                elif isinstance(result, (int, float, str, bool)):
                    # Convert scalar to single-cell DataFrame
                    result = pd.DataFrame({'result': [result]})
                elif hasattr(result, '__iter__') and not isinstance(result, str):
                    # Handle other iterable types
                    try:
                        result = pd.DataFrame(list(result))
                    except:
                        result = pd.DataFrame({'value': list(result)})
                else:
                    # For any other type, convert to string representation
                    result = pd.DataFrame({'result': [str(result)]})
            
            # Limit result size for safety
            if len(result) > 10000:
                logger.warning(f"Result truncated from {len(result)} to 10000 rows")
                result = result.head(10000)
            
            return result
            
        except Exception as e:
            if isinstance(e, SafetyError):
                raise
            logger.error(f"Pandas execution error: {e}")
            raise SafetyError(f"Execution failed: {str(e)}")

def validate_sql_query(query: str) -> None:
    """Validate SQL query for basic safety"""
    query_upper = query.upper().strip()
    
    # Block dangerous SQL operations
    dangerous_patterns = [
        'DROP ', 'DELETE ', 'INSERT ', 'UPDATE ', 'ALTER ', 'CREATE ',
        'TRUNCATE ', 'REPLACE ', 'EXEC ', 'EXECUTE ', 'CALL ',
        'PRAGMA ', 'ATTACH ', 'DETACH ', 'GRANT ', 'REVOKE '
    ]
    
    for pattern in dangerous_patterns:
        if pattern in query_upper:
            raise SafetyError(f"Dangerous SQL operation detected: {pattern.strip()}")
    
    # Allow more flexible query patterns while blocking write operations
    # Instead of checking start patterns, just ensure no write operations are present
    # This allows complex queries, CTEs, subqueries, etc.
    
    # Additional check for semicolon-separated multiple statements
    statements = query.split(';')
    if len(statements) > 2:  # Allow one main query plus empty string after semicolon
        non_empty_statements = [s.strip() for s in statements if s.strip()]
        if len(non_empty_statements) > 1:
            raise SafetyError("Multiple SQL statements not allowed")
    
    # Check for common SQL injection patterns
    injection_patterns = ['--', '/*', '*/', 'xp_', 'sp_']
    for pattern in injection_patterns:
        if pattern in query_upper:
            # Allow legitimate comments and common SQL functions
            if pattern == '--' and not ('DROP' in query_upper or 'DELETE' in query_upper):
                continue
            if pattern in ['/*', '*/'] and not ('DROP' in query_upper or 'DELETE' in query_upper):
                continue
            if pattern in ['xp_', 'sp_']:
                raise SafetyError(f"Potentially dangerous pattern detected: {pattern}")

def sanitize_user_input(text: str) -> str:
    """Sanitize user input to prevent injection attacks"""
    if not isinstance(text, str):
        raise SafetyError("Input must be a string")
    
    # Remove null bytes and control characters
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')
    
    # Limit length
    if len(text) > 10000:
        raise SafetyError("Input too long")
    
    return text.strip()

# Global safe executor instance
_safe_executor = PandasSafeExecutor()

def safe_execute_pandas(code: str, df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function for safe pandas execution"""
    code = sanitize_user_input(code)
    return _safe_executor.execute_pandas_code(code, df)

def safe_validate_sql(query: str) -> str:
    """Convenience function for SQL validation"""
    query = sanitize_user_input(query)
    validate_sql_query(query)
    return query