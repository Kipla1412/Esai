from src.agent.tool.factory import ToolFactory

def add_numbers(a: int,b:int) ->int:
    """
    Returns add  two numbers

    Args:

       a: first number
       b:second number

    Returns:
      the sum of a and b
    
    """
    return a+b

def multiply(a:int,b:int) ->int:
    """
    Returns multiply two numbers

    Args:

       a: first number
       b:second number

    Returns:
      the product of a and b
    
    """
    return a*b

config ={
    "tools": [
        add_numbers,
        multiply
        
    ]
}

tools = ToolFactory.create(config)

for t in tools:
    print("Tool name:", t.name)
    print("Description:", t.description)
    print("Inputs:", getattr(t, "inputs", None))
    print("---")

print("Add result:", tools[0].forward(5,7))
print("multiply result:",tools[1].forward(7,9))

