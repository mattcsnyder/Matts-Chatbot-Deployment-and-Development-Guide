import ollama

conversation_history = []

print("Chatbot ready. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ("exit", "quit"):
        break

    conversation_history.append({"role": "user", "content": user_input})
    response = ollama.chat(model="llama3.2", messages=conversation_history)

    bot_reply = response["message"]["content"]
    conversation_history.append({"role": "assistant", "content": bot_reply})

    print(f"Bot: {bot_reply}\n")
