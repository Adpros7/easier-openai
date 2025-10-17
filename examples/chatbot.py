from easier_aai import Assistant

assistant = Assistant(None, model="gpt-4.1-nano", system_prompt="You are a helpful assistant.")

while True:
    query = input("You: ")
    response = assistant.chat(query)
    print("Assistant:", response)