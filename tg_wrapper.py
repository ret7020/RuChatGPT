from aiogram import Bot, Dispatcher, executor, types
from bot import NeuralBot, UserSession

API_TOKEN = ''

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)
SESSIONS = {}

print("[DEBUG] Loading neural bot...")
neural_predictor = NeuralBot()


@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    await message.answer("Hello, lets talk to Russian GPT 3 based bot!")

@dp.message_handler(commands=["sessions"])
async def send_welcome(message: types.Message):
    user_id = message.from_user.id
    reply = "No sessions"
    if user_id in SESSIONS:
        if len(SESSIONS[user_id]) > 0:
            reply = f"Session id: {SESSIONS[user_id][0].session_id}"
    await message.answer(reply)

@dp.message_handler(commands=["clears"])
async def send_welcome(message: types.Message):
    user_id = message.from_user.id
    if user_id in SESSIONS:
        if len(SESSIONS[user_id]) > 0:
            del SESSIONS[user_id][0]
            SESSIONS[user_id] = []
    await message.answer("Sessions deleted")

@dp.message_handler(commands=["start_session"])
async def send_welcome(message: types.Message):
    user_id = message.from_user.id
    can_create = True
    if user_id in SESSIONS:
        if len(SESSIONS[user_id]) > 0:
            can_create = False
            await message.answer("Delete old sessions first")
    if can_create:
        await message.answer("🛠Starting job on creating new session...")
        SESSIONS[user_id] = [UserSession()]
        await message.answer("✅ Session started; You can chat now")

@dp.message_handler()
async def echo(message: types.Message):
    answered = False
    user_id = message.from_user.id
    if user_id in SESSIONS:
        if len(SESSIONS[user_id]) > 0:
            res = neural_predictor.answer("H", SESSIONS[user_id][0].chat_history_ids, message.text) # send message to bot
            SESSIONS[user_id][0].chat_history_ids = res[2] # save history to session
            answer = neural_predictor.answer("G", SESSIONS[user_id][0].chat_history_ids)
            SESSIONS[user_id][0].chat_history_ids = answer[2] # save history
            answered = True
            if answer[1]:
                await message.answer(answer[1])
            else:
                await message.answer("<-- БОТ РЕШИЛ ОТВЕТИТЬ ПУСТОТОЙ -->")
    if not answered:
        await message.answer("Create session first")






if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
