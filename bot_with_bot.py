from bot import NeuralBot, UserSession

print("[DEBUG] Creating base model")
predictor = NeuralBot()
print("[DEBUG] Creating 2 sessions")
session_bot_0 = UserSession()
session_bot_1 = UserSession()
bot_1_answer = "Привет"
print(f"1> {bot_1_answer}")
while True:
    # Send message to bot 0
    res = predictor.answer("H", session_bot_0.chat_history_ids, bot_1_answer)
    session_bot_0.chat_history_ids = res[2]
    # Bot 0 say
    res = predictor.answer("G", session_bot_0.chat_history_ids, None)
    # Save history to bot 0
    session_bot_0.chat_history_ids = res[2]
    print(f"0> {res[1]}")
    # Send message to bot 1
    res = predictor.answer("H", session_bot_1.chat_history_ids, res[1])
    # Save history for bot 1
    session_bot_1.chat_history_ids = res[2]
    # Bot 1 say
    res = predictor.answer("G", session_bot_1.chat_history_ids, None)
    session_bot_1.chat_history_ids = res[2]
    bot_1_answer = res[1]
    print(f"1> {bot_1_answer}")
    
