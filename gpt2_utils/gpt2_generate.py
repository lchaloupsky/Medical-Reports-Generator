#!/usr/bin/python3
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import argparse
import torch

parser = argparse.ArgumentParser()
# arguments used during generation of text
parser.add_argument('--max_len', default=512, type=int, help="Maximum length of generated text.")
parser.add_argument('--model', default="model_testing/gpt2_cz_med_smaller_full_16bs", type=str, help="Model name.")
parser.add_argument('--top_k', default=40, type=int, help="Take only top_k most probable tokens into account when generating.")
parser.add_argument('--top_p', default=0.95, type=float, help="Take only most probable tokens whose sum of probabilities is at most top_p when generating.")
parser.add_argument('--repetition_penalty', default=1.2, type=float, help="Repetition penalty.")
parser.add_argument('--temperature', default=0.7, type=float, help="Temperature for generating.")
parser.add_argument('--do_sample', default=True, type=bool, help="Do sample tokens when generating.")
parser.add_argument('--num_return_sequences', default=1, type=int, help="Number of sequences to be returned.")
parser.add_argument('--num_beams', default=None, type=int, help="Number of beams for search.")
parser.add_argument('--length_penalty', default=None, type=float, help="Exponential penalty for the text length.")
parser.add_argument('--bos_token_id', default=None, type=int, help="Beginning token id.")
parser.add_argument('--eos_token_id', default=None, type=int, help="End token id.")
parser.add_argument('--pad_token_id', default=50256, type=int, help="Pad token id.")

def generate(args: argparse.Namespace) -> None:
    if args.model == None:
        raise ValueError("The model argument must be set for generating the text!")

    # load given pretrained model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained(args.model).to(device)
    tokenizer_cs = GPT2Tokenizer.from_pretrained(args.model, pad_token='<|endoftext|>')

    # older versions !!!
    tokenizer_cs.max_len = args.max_len 

    while True:
        # read user prompt
        start = input()

        # if user entered empty prompt use special <|endoftext|> token
        if not start.strip():
            start = "<|endoftext|>"

        # older versions !!!
        input_ids = tokenizer_cs.encode(start, return_tensors='pt').to(device)
        sample_outputs = model.generate(
            input_ids, 
            pad_token_id=args.pad_token_id,
            eos_token_id=args.eos_token_id,
            bos_token_id=args.bos_token_id,
            do_sample=args.do_sample, 
            max_length=args.max_len,
            repetition_penalty=args.repetition_penalty,
            temperature=args.temperature, 
            top_k=args.top_k,
            top_p=args.top_p,
            num_return_sequences=args.num_return_sequences,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty)

        # older versions !!!
        for i, text in enumerate(sample_outputs if args.num_return_sequences == 1 else sample_outputs[0]):
            print(f"Generated text for input '{start}' {i+1}:\n\n{tokenizer_cs.decode(text, skip_special_tokens=True)}")

if __name__ == "__main__":
    generate(parser.parse_args([] if "__file__" not in globals() else None))