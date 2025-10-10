from easier_openai import Assistant

assistant = Assistant(
    api_key=None,
    model="gpt-3.5-turbo",
    system_prompt="You are a helpful assistant.",
)

while True:
    query = assistant.speech_to_text(mode="vad", model="gpt-4o-transcribe", aggressive=3)
    response = assistant.full_text_to_speech(query, voice="alloy", instructions="be calm", play=True, print_response=True)