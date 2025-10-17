from easier_aai import Assistant

assistant = Assistant(None, model="gpt-4o", system_prompt="You are a helpful assistant.")

while True:
    query = input("You: ")
    for response in assistant.chat(query, text_stream=True):
        if response == "done":
            break
        else:
            print("Assistant:", response)