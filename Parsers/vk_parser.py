import vk_api
import json
import time

vk_session = vk_api.VkApi(token='vk1.a.VduotujWOys5W6J64bWL1fp2RXJQKswf7757UMUNJqJKaku_lbhL3J9lbAn1d6AFqQ6AX6Z_W7YbXWlouJL7fGGwkwxmMMZc-JF65MmnNbFN3aC92PLdpxx1G8eQYxhvdjm3-msixcJIwsbpdlmSP9L2u31dm7pQqcNBesgototlnbv-f799Geixk4rUEdCMb6noIWJgpqO88IrVNssF7A')
vk = vk_session.get_api()
flag = True
res = []
chunks_counter = 0
with open("result_vk.json", "w", encoding="utf-8") as file:
    offset = 0
    while flag:
        message = vk.messages.getHistory(user_id=563797228, count=200, offset=offset)
        for message_count in range(200):
            if message_count >= message['count']:
                flag = False
                break
            message_text = message['items'][message_count]["text"]
            if message_text:
                from_ = "Степан Жданов" if message['items'][message_count]["from_id"] != 563797228 else "Михаил Легенов"
                res.append({"from": from_, "text": message_text})
        # save checkpoint
        json.dump(res, file)
        chunks_counter += 1
        print("Chunks saved:", chunks_counter, "Messages in chunk:", len(res)) 
        offset += 200
        time.sleep(0.3)
