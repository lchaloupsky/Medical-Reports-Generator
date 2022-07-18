#!/usr/bin/python3

# --- VERSION INFO
# Transformers 2.10.0 and lower
# Tokenizers 0.7.0 and lower

from fastai.text.all import *
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TFGPT2LMHeadModel
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2Config

import pandas as pd
import matplotlib.pyplot as plt
import sys
import argparse
import os

parser = argparse.ArgumentParser(description="Fine-tunes the new GPT-2 model based on the given arguments.")
# general arguments
parser.add_argument('--type', default='full', choices=['gradual', 'full'], type=str, help="Type of the finetuning method.")
parser.add_argument('--model', default="your_new_model_name", type=str, help="New model name.")
# training arguments
parser.add_argument('--pretrained_weights', default="gpt2", type=str, help="Pretrained model name.")
parser.add_argument('--dataset', default="oscar", type=str, help="Dataset name. Optional.")
parser.add_argument('--batch_size', default=16, type=int, help="Dataset name.")
parser.add_argument('--train_data_ratio', default=0.8, type=float, help="Portion of data to be used for trainig.")
parser.add_argument('--sequence_length', default=512, type=int, help="Dataset name.")
parser.add_argument('--debug', default=False, type=bool, help="Turns on debugging mode.")
parser.add_argument('--resume_training', default=True, type=bool, help="Resumes interrupted training from the newest checkpoint.")
parser.add_argument('--find_learning_rates', default=False, type=bool, help="Finds initial learning rates.")
parser.add_argument('--find_learning_rates_epoch', default=None, type=int, help="Epoch to find initial learning rates from.")
parser.add_argument('--pretokenize_data', default=True, type=bool, help="Pretokenizes whole dataset in advance.")
parser.add_argument('--save_checkpoints', default=True, type=bool, help="Saves all checkpoints during training.")
parser.add_argument('--learning_rates', default=[4e-3, 2e-3, 5e-4, 1e-4], nargs="+", type=float, help="Learning rates to use.")

BASE_PATH = "storage"
DATA_PATH = "data"
MODELS_PATH = "models"
TOKENIZERS_PATH = "tokenizers"
TRAINING_DATA_PATH = "training_data"
CHECKPOINTS_PATH = "checkpoints"

def train_tokenizer(config, dataset_name, args):
    '''
    Trains new tokenizer on the given data.

    :param config: Config object containing paths to all important model directories
    :param dataset_name: Dataset name
    :param args: Script arguments
    :return: (tok_en, tok_cs) Tuple of original tokenizer and newly trained tokenizer
    '''
    
    # tokenizer name and path declarations
    ByteLevelBPE_tokenizer_cs_rep = 'ByteLevelBPE_tokenizer_cs'
    path_to_ByteLevelBPE_tokenizer_cs_rep = config["tokenizers_path"]/ByteLevelBPE_tokenizer_cs_rep

    # load existing pretrained english gpt2 tokenizer 
    tokenizer_en = GPT2Tokenizer.from_pretrained(args.pretrained_weights)
    tokenizer_en.pad_token = tokenizer_en.eos_token
    if not args.resume_training or not any(config["tokenizers_path"].iterdir()):
        # train a Byte Level BPE (BBPE) tokenizer on the 'given data' by using the Tokenizers library (Hugging Face)
        # get GPT2 tokenizer_en vocab size
        ByteLevelBPE_tokenizer_cs_vocab_size = tokenizer_en.vocab_size

        # ByteLevelBPETokenizer Represents a Byte-level BPE
        ByteLevelBPE_tokenizer_cs = ByteLevelBPETokenizer()

        # get list of paths to corpus files and customize training with <|endoftext|> special GPT-2 token
        paths = [str(config["data_path"]/f'{dataset_name}_agg.txt')]
        ByteLevelBPE_tokenizer_cs.train(
            files=paths,
            vocab_size=ByteLevelBPE_tokenizer_cs_vocab_size, 
            min_frequency=2, 
            special_tokens=["<|endoftext|>"])

        # get sequence length max of 1024
        ByteLevelBPE_tokenizer_cs.enable_truncation(max_length=args.sequence_length)
        
        # save tokenizer
        if not (path_to_ByteLevelBPE_tokenizer_cs_rep).exists():
            path_to_ByteLevelBPE_tokenizer_cs_rep.mkdir(exist_ok=True, parents=True)

        # !!! older versions
        ByteLevelBPE_tokenizer_cs.save(str(path_to_ByteLevelBPE_tokenizer_cs_rep), ByteLevelBPE_tokenizer_cs_rep)
        os.replace(path_to_ByteLevelBPE_tokenizer_cs_rep/f"{ByteLevelBPE_tokenizer_cs_rep}-vocab.json", path_to_ByteLevelBPE_tokenizer_cs_rep/"vocab.json")
        os.replace(path_to_ByteLevelBPE_tokenizer_cs_rep/f"{ByteLevelBPE_tokenizer_cs_rep}-merges.txt", path_to_ByteLevelBPE_tokenizer_cs_rep/"merges.txt")
    else:
        print(f"Tokenizer for '{dataset_name}' has been already trained. Using existing version.")

    # import the tokenizer config files in Portuguese into the pre-trained GPT2 Tokenizer
    tokenizer_cs = GPT2Tokenizer.from_pretrained(str(path_to_ByteLevelBPE_tokenizer_cs_rep), pad_token='<|endoftext|>')
    
    # get sequence length max of 512
    # !!! - newer versions
    #tokenizer_cs.model_max_length = 512

    # !!! - older versions
    tokenizer_cs.max_len = args.sequence_length 
    #tokenizer_cs.enable_truncation(max_length=512)

    return tokenizer_en, tokenizer_cs

def compare_embedding_matrices(config, tokenizer_en, tokenizer_cs, args):
    '''
    Compares embedding matrices and updates the weight of the GPT-2 wte and lm_head accordingly.

    :param config: Config object containing paths to all important model directories
    :param tokenizer_en: Original tokenizer
    :param tokenizer_cs: New tokenizer
    :param args: Script arguments
    :return: Updated GPT-2 model
    '''
    tokenizer_fastai_en = TransformersTokenizer(tokenizer_en)
    tokenizer_fastai_cs = TransformersTokenizer(tokenizer_cs)

    # load pretrained model
    model = GPT2LMHeadModel.from_pretrained(args.pretrained_weights)

    # get weights of the old wte
    old_wgts = model.transformer.get_input_embeddings().weight.clone().detach()

    # get the mean embedding vector of the old wte
    wgts_m = old_wgts.mean(0)

    # initialize vocab size and weights of the new wte
    new_vocab_size = tokenizer_fastai_cs.tokenizer.vocab_size
    new_wgts = old_wgts.new_zeros(new_vocab_size, old_wgts.size(1))

    # - get the new 'wte' keeping the embedding vectors of tokens in common in the 2 vocabs
    # - a token present in the new vocab but not in the old one gets the mean embedding vector of the old wte
    old_vocab = tokenizer_fastai_en.tokenizer.encoder # !!! older versions (newer have get_vocab() instead of encoder)
    new_vocab = tokenizer_fastai_cs.tokenizer.encoder # !!! older versions (newer have get_vocab() instead of encoder)
    same_tokens_list, different_tokens_list = list(), list()
        
    # compare the vocabs
    for w, idx_new in new_vocab.items():    
        idx_old = old_vocab.get(w, -1)
        if idx_old >= 0:
            new_wgts[idx_new] = old_wgts[idx_old]
            same_tokens_list.append((w, idx_new))
        else:
            new_wgts[idx_new] = wgts_m
            different_tokens_list.append((w, idx_new))

    # setup in model the new 'wte'
    new_wte = nn.Embedding(new_vocab_size, old_wgts.size(1))
    new_wte.weight.data = new_wgts
    model.transformer.set_input_embeddings(new_wte)

    # save new_wgts of 'wte'
    torch.save(new_wgts, config["training_data_path"]/'new_wte_wgts.pt')

    # save same_tokens_list and different_tokens_list
    torch.save(same_tokens_list, config["training_data_path"]/'same_tokens_list.pt')
    torch.save(different_tokens_list, config["training_data_path"]/'different_tokens_list.pt')

    # changing 'lm_head' weights with the new embedding
    model.lm_head.weight = model.transformer.wte.weight

    return model

def load_dataframe(config, dataset_name, args):
    '''
    Loads dataset and creates training and validation indices.

    :param config: Config object containing paths to all important model directories
    :param dataset_name: Dataset name
    :param args: Script arguments
    :return: data frame, train indices, validation indices
    '''
    df = pd.read_csv(config["data_path"]/f'{dataset_name}_agg.csv')

    # For debug purposes take only first 50 documents
    if args.debug:
        df = df.head(50)

    # print dataframe before and after filtering
    print(df, flush=True)
    df.dropna(inplace=True)
    print(df, flush=True)
    
    # train, dev split
    num = int(args.train_data_ratio * len(df))
    # - ORIGINAL - this can take some items multiple times
    #idxs = np.random.randint(0, len(df), len(df))
    # - NEW - this takes every item
    idxs = np.arange(len(df)) if args.debug else np.random.choice(len(df), len(df), replace=False)

    idxs_train = idxs[:num]
    idxs_val = idxs[num:]

    return df, idxs_train, idxs_val

def show_batch_text(batch, tokenizer, column_size: int = 75, show_by_words: bool = True) -> str:
    '''
    Shows all batch data from dataloaders.show_batch(...) in the text form instead of the html form.

    :param batch: Batch of data to be showed
    :param tokenizer: Corresponding text tokenizer
    :param column_size: Width of the column, defaults to 75
    :param show_by_words: Flag whether the column should be matched by number of characters or words, defaults to True
    :return: Formatted batch data texts
    '''
    delim = " | "
    inputs, validations, n = batch[0], batch[1], len(batch[2])

    # header
    final = f"{'inputs':>{column_size}}{' ' * len(delim)}{'validations':>{column_size}}\n"
    final += f"{'-' * (2 * column_size + len(delim))}\n"

    # body
    for i, v in zip(inputs[:n], validations[:n]):
        i_d, v_d = repr(tokenizer.decode(i)).strip("'"), repr(tokenizer.decode(v)).strip("'")        
        
        # character version
        if not show_by_words:
            num_blocks = (len(i_d) // column_size) + (1 if len(i_d) % column_size > 0 else 0)
            for i in range(num_blocks):
                start = i * column_size
                final += f"{i_d[start : start + column_size]:>{column_size}}{delim}{v_d[start : start + column_size]:>{column_size}}\n"
        # word version
        else:
            final_i, final_v = "", ""
            i_split, v_split = i_d.split(" "), v_d.split(" ") 
            for i2 in range(max(len(i_split), len(v_split))):
                w1 = "" if i2 >= len(i_split) else i_split[i2]
                w2 = "" if i2 >= len(v_split) else v_split[i2]
                if len(final_i) + len(w1) + 1 <= column_size and len(final_v) + len(w2) + 1 <= column_size:
                    final_i += f" {w1}" if i2 > 0 and i2 < len(i_split) else w1
                    final_v += f" {w2}" if i2 > 0 and i2 < len(v_split) else w2
                    continue

                final += f"{final_i:>{column_size}}{delim}{final_v:>{column_size}}\n"
                final_i, final_v = f" {w1}" if i2 < len(i_split) else "", f" {w2}" if i2 < len(v_split) else ""

            final += f"{final_i:>{column_size}}{delim}{final_v:>{column_size}}\n"

        final += "\n"

    return final


### ----------------- TOKENIZE AT RUNTIME VERSION -----------------------
class TransformersTokenizer(Transform):
    '''Tokenizer wrapper for the fastai process.'''
    def __init__(self, tokenizer): self.tokenizer = tokenizer
    def encodes(self, x): 
        toks = self.tokenizer.tokenize(x)
        return tensor(self.tokenizer.convert_tokens_to_ids(toks))
    def decodes(self, x): return TitledStr(self.tokenizer.decode(x.cpu().numpy()))

def prepare_data_without_tokenization(df, idxs_train, idxs_val, tokenizer_cs, args):
    '''
    Prepares data to corresponding structures without pre-tokenizing them in advance.

    :param df: Dataset DataFrame
    :param idxs_train: Train indices
    :param idxs_val: Validation indices
    :param tokenizer_cs: New tokenizer
    :param args: Script arguments
    :return: Prepared DataLoaders object
    '''

    # gather all texts in one numpy array 
    all_texts = np.concatenate([df.iloc[idxs_train].text.values, df.iloc[idxs_val].text.values])

    # create a pipeline which applies transformations to a data
    tls = TfmdLists(all_texts, TransformersTokenizer(tokenizer_cs), splits=[list(idxs_train), list(idxs_val)], dl_type=LMDataLoader)

    # print data for verification
    # -- encoded
    print(tls.train[0])
    # -- decoded
    print(show_at(tls.train, 0))

    # create dataloaders object for our data
    dls = tls.dataloaders(bs=args.batch_size, seq_len=args.sequence_length, num_workers=32, shuffle=False, pin_memory=True)

    # show batch data - 'input' and 'desired output'
    data_html = dls.show_batch(max_n=5)
    data_html = show_batch_text(data_html, tokenizer_cs)
    print(data_html)
    
    return dls
### ---------------------------------------------------------------------

### ---------------- PRETOKENIZE WHOLE DATASET VERSION ------------------
class TransformersTokenizerPreTokenized(Transform):
    '''Tokenizer wrapper for the fastai process.'''
    def __init__(self, tokenizer): 
        self.tokenizer = tokenizer

    def encodes(self, x): 
        return x if isinstance(x, Tensor) else tokenize(x, self.tokenizer)
        
    def decodes(self, x): 
        return TitledStr(self.tokenizer.decode(x.cpu().numpy()))

def tokenize(text, tokenizer_cs):
    '''
    Tokenizes the given text into token ids in the tensor.

    :param text: Text to be tokenized
    :param tokenizer_cs: Tokenizer object
    :return: Tokenized text tensor
    '''
    toks = tokenizer_cs.tokenize(text)
    return tensor(tokenizer_cs.convert_tokens_to_ids(toks))

def prepare_data_with_tokenization(config, df, idxs_train, idxs_val, tokenizer_cs, dataset_name, args):
    '''
    Prepares data to corresponding structures while pre-tokenizing them in advance.

    :param config: Config object containing paths to all important model directories
    :param df: Dataset DataFrame
    :param idxs_train: Train indices
    :param idxs_val: Validation indices
    :param tokenizer_cs: New tokenizer
    :param dataset_name: Dataset name
    :param args: Script arguments
    :return: Prepared DataLoaders object
    '''

    # gather all texts in one numpy array 
    all_texts = np.concatenate([df.iloc[idxs_train].text.values, df.iloc[idxs_val].text.values])

    # do tokenization of all texts in advance
    # tokenize all_texts only if there were not yet tokenized
    tokenized_file_name = f"tokenized_{dataset_name}_gpt2.pt"
    if not os.path.exists(str(config["training_data_path"]/tokenized_file_name)):
        print(f"Tokenizing '{dataset_name}' dataset.")

        # tokenize all texts
        tokenized = [tokenize(t, tokenizer_cs) for t in all_texts]

        # save and reload
        torch.save(tokenized, config["training_data_path"]/tokenized_file_name)
        tokenized_cs = torch.load(config["training_data_path"]/tokenized_file_name)
    else:
        print(f"Dataset '{dataset_name}' is already tokenized, loading from file.")
        tokenized_cs = torch.load(config["training_data_path"]/tokenized_file_name)

    # create a pipeline which applies transformations to a data
    tls2 = TfmdLists(
        tokenized_cs, TransformersTokenizerPreTokenized(tokenizer_cs), 
        splits=[list(idxs_train), list(idxs_val)], dl_type=LMDataLoader)

    # print data for verification
    # -- encoded
    print(tls2.train[0])
    # -- decoded
    print(show_at(tls2.train, 0))

    # create dataloaders object for our data
    dls = tls2.dataloaders(bs=args.batch_size, seq_len=args.sequence_length)

    # show batch data - 'input' and 'desired output'
    data_html = dls.show_batch(max_n=5, show=False)
    data_html = show_batch_text(data_html, tokenizer_cs)
    print(data_html)

    with open(str(config["training_data_path"]/"batch_data.txt"), "w") as file:
        file.write(data_html)

    return dls
### ---------------------------------------------------------------------

class DropOutput(Callback):
    def after_pred(self):
        # Because GPT2 returns a multiple tensors, we need to keep only the predictions 
        # The rest of it are additional activations, which can be used for some regularization
        # Note: self.pred is just a READ ONLY shortcut for self.lear.pred 
        self.learn.pred = self.pred[0]

class StartFromEpoch(Callback):
    '''Callback to be able to skip some epochs and train later phases.'''
    def __init__(self, start_epoch):
        self.start_epoch = start_epoch

    def before_train(self):
        if self.epoch < self.start_epoch: 
            raise CancelEpochException

    def before_validate(self):
        if self.epoch < self.start_epoch: 
            raise CancelValidException

def splitter(n=4):
    '''
    Splits a GPT2 model layers in 4 groups for differential learning rates:
    (1) - first 'n' decoder blocks
    (2) - second 'n' decoder blocks
    (3) - third 'n' decoder blocks
    (4) - wte, wpe and final LayerNorm
    
    :param n: Number of blocks to split by
    :return: Function 'inner_splitter' that splits the blocks into groups
    '''
    def inner_splitter(model):
        # First group: decoder blocks from 0 to 'n-1'
        modules = []
        for i in range(n): 
            modules.append(model.transformer.h[i])
        groups = [nn.Sequential(*modules)]    
        
        # Second group: decoder blocks from 'n' to '2n-1'
        modules = []
        for i in range(n, 2*n): 
            modules.append(model.transformer.h[i])
        groups = L(groups + [nn.Sequential(*modules)])    
        
        # Third group: decoder blocks from '2n' to '3n-1'
        modules = []
        for i in range(2*n, 3*n): 
            modules.append(model.transformer.h[i])
        groups = L(groups + [nn.Sequential(*modules)])
        
        # Fourth group: embedding matrices wte and wpe + LayerNorm at the model output
        groups = L(groups + [nn.Sequential(model.transformer.wte, model.transformer.wpe, model.transformer.ln_f)])
        
        return groups.map(params)

    return inner_splitter

def find_learning_rates(training_data_path, learn, exit=True, name="lr_plot", epoch=None):
    '''
    Finds and saves learning rates to be used.

    :param training_data_path: Path to the training data
    :param learn: Learner object
    :param exit: If the program should exit after finding the learning rates, defaults to True
    :param name: Name of the created learning rate plot, defaults to "lr_plot"
    :param epoch: Epoch in which the learning rates were found, defaults to None
    '''

    # Find all possible good learning rates
    lrs = learn.lr_find(show_plot=True, suggest_funcs=(minimum, steep, valley, slide))

    # Print them and save the loss plot
    print(lrs.minimum, lrs.steep, lrs.valley, lrs.slide)
    plt.savefig(f"{str(training_data_path/name)}{'' if epoch is None else f'_e{str(epoch)}'}.png")

    if exit: sys.exit()

def finetune_gradual(model_dirs, model_name, learn, tokenizer_cs, dataset_name, args):
    '''
    Fine-tunes the model in the gradual way.

    :param model_dirs: Model directories tuple - Final model destination, Checkpoints destination, Training data destination
    :param model_name: New model name
    :param learn: Learner object
    :param tokenizer_cs: New tokenizer
    :param dataset_name: Dataset name
    :param args: Script arguments
    '''
    final_dest, checkpoint_dest, training_data_dest = model_dirs

    # --- FINE-TUNING ---
    if not any(checkpoint_dest.iterdir()) or not args.resume_training:
        # Freeze whole model except 'wte', 'wpe', 'LayerNorm' and final 'Linear' layer
        learn.freeze()
        print(learn.summary())
        # find learning rates
        if args.find_learning_rates and args.find_learning_rates_epoch is None or args.find_learning_rates_epoch == 0:
            find_learning_rates(training_data_dest, learn, epoch=-1)

        learn.fit_one_cycle(1, args.learning_rates[0])
        save_checkpoint(learn, f'{model_name}_{dataset_name}_lr{args.learning_rates[0]:.2e}_e_-1', args)
        plot_and_save_loss(learn, training_data_dest, "loss_1epoch")

    # load last checkpoint
    if args.resume_training:
        learn, epoch = load_checkpoint(learn, checkpoint_dest)

    # check it is not the very last checkpoint
    if epoch >= 3:
        return

    for i in range(1, 3):
        if epoch + 1 >= i:
            continue

        # Unfreeze last '(i+1)*n' decoder blocks
        learn.freeze_to(-(i + 1))
        print(learn.summary())
        if args.find_learning_rates and args.find_learning_rates_epoch == i:
            find_learning_rates(training_data_dest, learn, epoch=i-1)

        learn.fit_one_cycle(1, slice(args.learning_rates[i]/(2.6**4), args.learning_rates[i]))
        save_checkpoint(learn, f'{model_name}_{dataset_name}_lr{args.learning_rates[i]:.2e}_e_{i-1}', args)
        plot_and_save_loss(learn, training_data_dest, f"loss_{i+1}epoch")

    # Unfreeze whole model
    learn.unfreeze()
    print(learn.summary())
    if args.find_learning_rates and args.find_learning_rates_epoch >= 3:
        find_learning_rates(training_data_dest, learn, epoch="2_3")

    learn.fit_one_cycle(2, slice(args.learning_rates[3]/(2.6**4), args.learning_rates[3]))
    save_checkpoint(learn, f'{model_name}_{dataset_name}_lr{args.learning_rates[3]:.2e}_len2_e_3', args)
    plot_and_save_loss(learn, training_data_dest, "loss_4_5epoch")

    # --- SAVING THE MODEL ----
    save_models(final_dest, learn, tokenizer_cs)

def finetune_all_at_once(model_dirs, model_name, learn, tokenizer_cs, dataset_name, args):
    '''
    Fine-tunes the model in two phases - new head only, entire model.

    :param model_dirs: Model directories tuple - Final model destination, Checkpoints destination, Training data destination
    :param model_name: New model name
    :param learn: Learner object
    :param tokenizer_cs: New tokenizer
    :param dataset_name: Dataset name
    :param args: Script arguments
    '''
    final_dest, checkpoint_dest, training_data_dest = model_dirs

    # --- FINE-TUNING ---
    if not any(checkpoint_dest.iterdir()) or not args.resume_training:
        # Freeze whole model except 'wte', 'wpe', 'LayerNorm' and final 'Linear' layer
        learn.freeze()
        print(learn.summary())
        # find learning rates
        if args.find_learning_rates and args.find_learning_rates_epoch is None or args.find_learning_rates_epoch == 0:
            find_learning_rates(training_data_dest, learn, epoch=-1)

        learn.fit_one_cycle(1, args.learning_rates[0])
        save_checkpoint(learn, f'{model_name}_{dataset_name}_lr{args.learning_rates[0]:.2e}_e_-1', args)
        plot_and_save_loss(learn, training_data_dest, "loss_1epoch")

    if args.resume_training:
        learn, epoch = load_checkpoint(learn, checkpoint_dest)

    # Unfreeze whole model
    learn.unfreeze()
    print(learn.summary())
    if args.find_learning_rates and args.find_learning_rates_epoch >= 1:
        find_learning_rates(training_data_dest, learn, epoch="0_3")

    callbacks = [SaveModelCallback(fname=model_name, every_epoch=True, with_opt=True)] + ([StartFromEpoch(start_epoch=epoch + 1)] if args.resume_training else [])
    learn.fit_one_cycle(4, 
        slice(args.learning_rates[1]/(2.6**4), args.learning_rates[1]), 
        cbs=callbacks
    )
    save_checkpoint(learn, f'{model_name}_{dataset_name}_lr{args.learning_rates[1]:.2e}_len4_e_3', args)
    plot_and_save_loss(learn, training_data_dest, "loss_2_5epoch")

    learn.recorder.plot_sched()
    plt.savefig(training_data_dest/f"lrs.png")

    # --- SAVING THE MODEL ----
    save_models(final_dest, learn, tokenizer_cs)

def save_models(final_dest, learn, tokenizer_cs):
    '''
    Saves model.

    :param final_dest: Destination where to save to model
    :param learn: Learner object
    :param tokenizer_cs: Tokenizer object
    '''
    # save pyTorch version of the trained model
    learn.model.save_pretrained(str(final_dest))
    tokenizer_cs.save_pretrained(str(final_dest))
    print("Saved 'pyTorch' version.")

    # save tensorflow version of the trained model
    tf_model = TFGPT2LMHeadModel.from_pretrained(str(final_dest), from_pt=True)
    tf_model.save_pretrained(str(final_dest))
    print("Saved 'tensorflow' version.")

def load_checkpoint(learn, checkpoints_dir, name=None):
    '''
    Loads existing checkpoint.

    :param learn: Learner object
    :param checkpoints_dir: Checkpoints directory
    :param name: Name of the checkpoint - optional, defaults to None
    :return: Loaded checkpoint Learner object
    '''
    if name is not None:
        return learn.load(name)
    
    latest_ckpt = max(checkpoints_dir.glob(r"*.pth"), key=lambda f: f.stat().st_ctime).name
    return learn.load(".".join(latest_ckpt.split(".")[:-1])), int(latest_ckpt.split("_")[-1].split(".")[0])

def prepare_model_dirs(config, args, dataset_name):
    '''
    Prepares all dirs for the current model.

    :param config: Config object containing paths to all important model directories
    :param args: Script arguments
    :param dataset_name: Dataset name
    :return: Tuple of all neccessary directories and model name - ((model_dir, checkpoints_dir, training_data_dir), model_name)
    '''

    # Prepares all dirs for the current model
    model_name = args.model if args.model is not None else f"{args.pretrained_weights}_{dataset_name}_{'_'.join(args.learning.rates)}"
    final_dest = config["models_path"]/model_name
    checkpoints_dest = config["checkpoints_path"]/model_name
    training_data_dest = config["training_data_path"]/model_name

    # Create dirs for the specific model
    final_dest.mkdir(exist_ok=True)
    checkpoints_dest.mkdir(exist_ok=True)
    training_data_dest.mkdir(exist_ok=True)

    return (final_dest, checkpoints_dest, training_data_dest), model_name

def save_checkpoint(learn, checkpoint_name, args):
    '''
    Saves checkpoint.

    :param learn: Learner object
    :param checkpoint_name: Checkpoint name
    :param args: Script arguments
    '''
    if args.save_checkpoints:
        learn.save(checkpoint_name)

def plot_and_save_loss(learn, destination, plot_name):
    '''
    Plots and saves last run training losses

    :param learn: Learner object
    :param destination: Plot destination
    :param plot_name: Plot name
    '''

    learn.recorder.plot_loss()
    plt.savefig(f"{str(destination/plot_name)}.png")
    plt.clf()

def prepare_config(dataset_name):
    '''
    Prepares all basic directories for given dataset.

    :param dataset_name: Dataset name
    :return: Prepared config object - dictionary of "data_path", "tokenizers_path", "training_data_path", "models_path" and "checkpoints_path"
    '''
    base_path = Path(__file__).parent.resolve()/BASE_PATH
    config = {
        "data_path": base_path/DATA_PATH/dataset_name,
        "tokenizers_path": base_path/TOKENIZERS_PATH/dataset_name,
        "training_data_path": base_path/TRAINING_DATA_PATH/dataset_name,
        "models_path": base_path/MODELS_PATH/dataset_name,
        "checkpoints_path": base_path/CHECKPOINTS_PATH/dataset_name
    }

    for v in config.values():
        v.mkdir(exist_ok=True, parents=True)

    return config

def train(args: argparse.Namespace) -> None:
    '''
    Main entry of the script, where the entire training process runs.

    :param args: Script arguments
    '''

    # check pretrained model config
    print(GPT2Config.from_pretrained(args.pretrained_weights))

    # create folders where all data will be stored
    dataset_name = args.dataset
    config = prepare_config(dataset_name)
    print(config)

    # Main pipeline for preparing data, tokenizer and model
    tokenizer_en, tokenizer_cs = train_tokenizer(config, dataset_name, args)
    model = compare_embedding_matrices(config, tokenizer_en, tokenizer_cs, args)
    df, idxs_train, idxs_val = load_dataframe(config, dataset_name, args)
    if args.pretokenize_data:
        dls = prepare_data_with_tokenization(config, df, idxs_train, idxs_val, tokenizer_cs, dataset_name, args)
    else:
        dls = prepare_data_without_tokenization(df, idxs_train, idxs_val, tokenizer_cs, args)

    # Final step - finetuning the gpt2
    # note: to_fp16() turns on the half precision evaluation - uses less GPU memory
    model_dirs, model_name = prepare_model_dirs(config, args, dataset_name)
    learn = Learner(
        dls, model, 
        loss_func=CrossEntropyLossFlat(),
        splitter=splitter(n=len(model.transformer.h) // 3),
        cbs=[DropOutput], 
        metrics=[accuracy, Perplexity()],
        path=".",
        model_dir=model_dirs[1]
    ).to_fp16()

    if args.type == "gradual": 
        finetune_gradual(model_dirs, model_name, learn, tokenizer_cs, dataset_name, args)
    else: 
        finetune_all_at_once(model_dirs, model_name, learn, tokenizer_cs, dataset_name, args)

if __name__ == "__main__":
    train(parser.parse_args([] if "__file__" not in globals() else None))