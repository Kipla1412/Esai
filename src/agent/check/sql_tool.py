
import os
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, Float, inspect, text, insert

engine = create_engine("sqlite:/// text_db_sql.db")
metadata_obj = MetaData()

table_name = "products"

products_table= Table(
    table_name,metadata_obj,
    Column("product_id",Integer,primary_key=True),
    Column("product_name",String),
    Column("category",String),
    Column("price",Float)
)

metadata_obj.create_all(engine)

# connect or insert the sample datas

conn = engine.connect()
with conn.begin() as trans:

    conn.execute(insert(products_table),[
        {'product_name': 'Laptop', 'category': 'Electronics', 'price': 1200.00},
        {'product_name': 'Desk Chair', 'category': 'Furniture', 'price': 250.50},
        {'product_name': 'T-shirt', 'category': 'Apparel', 'price': 25.00},
        {'product_name': 'Smartphone', 'category': 'Electronics', 'price': 800.00},
        {'product_name': 'Coffee Table', 'category': 'Furniture', 'price': 150.75}
    ])

    trans.commit()

def execute_sql(query: str) ->str:
    """
    Executes a SQL query against the connected database.

    This tool is designed to be called by an agent to run a valid SQL query
    and retreive the results. It handles basic error management and returns a formatted
    string containing and outcome.

    Args:

        query: The SQL query string to be executed against the database.

    Returns:
        A string containing the query results in a list of dictionaries if successful,
        or a formatted error message if the query fails.
       
    """
    try:

        with engine.connect() as conn:
            result =conn.execute(text(query))

            rows =[row._asdict() for row in result.fetchall()]

        return f"Query executed sucessfully. Result:{rows}"
    except Exception as e:
        return f"Error executing SQL query. Error: {e}"
    
inspector = inspect(engine)
table_names =inspector.get_table_names()

db_schema = "tables:\n" + "\n".join(
    f"  - {table_name}: columns: {', '.join([col['name'] for col in inspector.get_columns(table_name)])}"
    for table_name in table_names
)







