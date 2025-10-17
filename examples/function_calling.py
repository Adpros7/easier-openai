from easier_ai import Assistant, assistant

assistant = Assistant(None, model="gpt-4o", system_prompt="You are a helpful assistant.")

def say_hi(name: str):
    """Says hi to a person.
    Parameters:
        name (str): The name of the person to say hi to.
    """
    print(f"Hi {name}")
    
print(assistant.chat("say hi to bob", custom_tools=[say_hi]))