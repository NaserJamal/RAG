"""
SQL Tools - Execute SQL queries using SQLAlchemy.
"""

import os
from sqlalchemy import create_engine, text
from core.tool_system import registry


@registry.register(
    name="execute_sql",
    description="Execute a SQL query and return the results",
    parameters={
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {
                "type": "string",
                "description": "The SQL query to execute"
            }
        }
    }
)
def execute_sql(query: str) -> dict:
    """
    Execute a SQL query against the configured database.

    Args:
        query: The SQL query to execute

    Returns:
        A dictionary containing the query results or error information
    """
    try:
        # Get database URL from environment
        db_url = os.getenv("SQL_TOOL_DATA_BASE_URL")
        if not db_url:
            return {
                "error": "SQL_TOOL_DATA_BASE_URL environment variable not set"
            }

        # Create engine and execute query
        engine = create_engine(db_url)
        with engine.connect() as connection:
            result = connection.execute(text(query))

            # Handle SELECT queries (return rows)
            if result.returns_rows:
                rows = [dict(row._mapping) for row in result]
                return {
                    "rows": rows,
                    "row_count": len(rows)
                }
            # Handle INSERT/UPDATE/DELETE queries
            else:
                connection.commit()
                return {
                    "affected_rows": result.rowcount,
                    "message": "Query executed successfully"
                }

    except Exception as e:
        return {
            "error": str(e)
        }
