# import pandas as pd
# from .sql_tool import engine
# from sqlalchemy import text


# def sql_agent(query: str) ->str:
#     """
#     Allows you to perform SQL queries on the table. Returns a string representation of the result.

#     Args:
#        query: user asked the question from the database.

#     Returns:

#         Query Results as string or error message

#     """
#     if not query.strip():

#         return "query is empty."
#     try:
#         output =""
#         with  engine.connect() as con:

#             result = con.execute(text(query))
#             rows = result .fetchall()

#             if not rows:
#                 return "no matching records found."
            
#             for row in rows:
#                 output += "\n" + ", ".join(str(col) for col in row)


#         return output.strip()
    
#     except Exception as e:
#         return f"ERROR while executing SQL: {str(e)}"






