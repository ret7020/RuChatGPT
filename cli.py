import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from base import *
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./model.pt", type=str, help="Path to model weights")
    args = parser.parse_args()
    # Load base model
    checkpoint = "Kirili4ik/ruDialoGpt3-medium-finetuned-telegram"   
    tokenizer =  AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint)

    # Load finetuned weights
    checkpoint = torch.load(args.model, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to('cpu') # Perfectly works on cpu machine; My nvidia can't fit this model;
    model.eval()

    chat_history_ids = torch.zeros((1, 0), dtype=torch.int)
    next_who = "H"
    next_len = 2
    while True:
        if next_who == "H":
            input_user = input("Human> ")
            if input_user == "/next":
                next_who = "G"
                continue
            # encode the new user input, add parameters and return a tensor in Pytorch
            new_user_input_ids = tokenizer.encode(f"|0|{get_length_param(input_user, tokenizer)}|" \
                                                + input_user + tokenizer.eos_token, return_tensors="pt")
            # append the new user input tokens to the chat history
            chat_history_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
            next_who = "G"

        if next_who == "G":
            # encode the new user input, add parameters and return a tensor in Pytorch
            new_user_input_ids = tokenizer.encode(f"|1|{next_len}|", return_tensors="pt")
            # append the new user input tokens to the chat history
            chat_history_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
            
            # print(tokenizer.decode(chat_history_ids[-1])) # uncomment to see full gpt input
            
            # save previous len
            input_len = chat_history_ids.shape[-1]
            # generated a response; PS you can read about the parameters at hf.co/blog/how-to-generate
            chat_history_ids = model.generate(
                chat_history_ids,
                num_return_sequences=1,                     # use for more variants, but have to print [i]
                max_length=2048,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                temperature = 0.6,                          # 0 for greedy
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            
            # pretty print last ouput tokens from bot
            print(f"Bot> {tokenizer.decode(chat_history_ids[:, input_len:][0], skip_special_tokens=True)}")
            next_who = "H"



