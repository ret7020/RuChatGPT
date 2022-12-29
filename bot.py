import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from base import *
import random

class NeuralBot:
    def __init__(self, model_path="./joined.pt"): # Create session
        checkpoint = "Kirili4ik/ruDialoGpt3-medium-finetuned-telegram"   
        self.tokenizer =  AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint)
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to('cpu') # Perfectly works on cpu machine; My nvidia can't fit this model;
        self.model.eval()
        self.ans_len = 3
        self.session_id = random.randint(10000, 10000000)

    def answer(self, next_who, history, input_user=None):
        chat_history_ids = history
        if next_who == "H":
            if input_user == "/next":
                return self.answer("G", chat_history_ids)
            # encode the new user input, add parameters and return a tensor in Pytorch
            new_user_input_ids = self.tokenizer.encode(f"|0|{get_length_param(input_user, self.tokenizer)}|" \
                                                + input_user + self.tokenizer.eos_token, return_tensors="pt")
            # append the new user input tokens to the chat history
            chat_history_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
            return ("G", None, chat_history_ids)

        if next_who == "G":
            # encode the new user input, add parameters and return a tensor in Pytorch
            new_user_input_ids = self.tokenizer.encode(f"|1|{self.ans_len}|", return_tensors="pt")
            # append the new user input tokens to the chat history
            chat_history_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
            
            # print(tokenizer.decode(chat_history_ids[-1])) # uncomment to see full gpt input
            
            # save previous len
            input_len = chat_history_ids.shape[-1]
            # generated a response; PS you can read about the parameters at hf.co/blog/how-to-generate
            chat_history_ids = self.model.generate(
                chat_history_ids,
                num_return_sequences=1,                     # use for more variants, but have to print [i]
                max_length=2048,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                temperature = 0.6,                          # 0 for greedy
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            
            # pretty print last ouput tokens from bot
            #print(f"Bot> {self.tokenizer.decode(self.chat_history_ids[:, input_len:][0], skip_special_tokens=True)}")
            return ("H", self.tokenizer.decode(chat_history_ids[:, input_len:][0], skip_special_tokens=True), chat_history_ids)
class UserSession:
    def __init__(self):
        self.session_id = random.randint(10000, 100000)
        self.chat_history_ids = torch.zeros((1, 0), dtype=torch.int)
        self.next_who = "H"

if __name__ == "__main__":
    print("[DEBUG] Creating base model")
    base = NeuralBot()
    print("[DEBUG] Creating session")
    session = UserSession()
    while True:
        data = None
        if session.next_who == "H":
            data = input("H>")
        res = base.answer(session.next_who, session.chat_history_ids, data)
        session.chat_history_ids = res[2]
        session.next_who = res[0]
        if res[1] is not None:
            print(res[1])
