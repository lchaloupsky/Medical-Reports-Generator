#!/usr/bin/python3

import tensorflow as tf
import numpy as np

from CNN_encoder import CNN_Encoder
from configs import argHandler
from tokenizer_wrapper import TokenizerWrapper
from PIL import Image
from gpt2.gpt2_model import TFGPT2LMHeadModel
from skimage.transform import resize

def load_image(image_file):
    print(f"Loading '{image_file}'")
    image = Image.open(image_file)
    image_array = np.asarray(image.convert("RGB"))
    image_array = image_array / 255.
    image_array = np.asarray([resize(image_array, (224, 224, 3))])
    
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    
    image_array = (image_array - imagenet_mean) / imagenet_std
    return image_array

def predict(FLAGS, encoder, decoder, tokenizer_wrapper, image):
    visual_features, tags_embeddings = encoder(image, training=False)
    dec_input = tf.expand_dims(tokenizer_wrapper.GPT2_encode("startseq", pad=False), 0)
    num_beams = FLAGS.beam_width

    visual_features = tf.tile(visual_features, [num_beams, 1, 1])
    tags_embeddings = tf.tile(tags_embeddings, [num_beams, 1, 1])
    tokens = decoder.generate(dec_input, max_length=FLAGS.max_sequence_length, num_beams=num_beams, min_length=3,
                              eos_token_ids=tokenizer_wrapper.GPT2_eos_token_id(), no_repeat_ngram_size=0,
                              visual_features=visual_features,
                              tags_embedding=tags_embeddings, do_sample=False, early_stopping=True)

    sentence = tokenizer_wrapper.GPT2_decode(tokens[0])
    sentence = tokenizer_wrapper.filter_special_words(sentence)
    return sentence

if __name__ == "__main__":
    FLAGS = argHandler()
    FLAGS.setDefaults()

    # prepare tokenizer
    tokenizer_wrapper = TokenizerWrapper(FLAGS.all_data_csv, FLAGS.csv_label_columns[0], FLAGS.max_sequence_length, FLAGS.tokenizer_vocab_size)

    # prepare objects from checkpoint
    encoder = CNN_Encoder('pretrained_visual_model', FLAGS.visual_model_name, FLAGS.visual_model_pop_layers, FLAGS.encoder_layers, FLAGS.tags_threshold, num_tags=len(FLAGS.tags))
    decoder = TFGPT2LMHeadModel.from_pretrained('gpt2_cz_oscar_fulldata_full_16bs')
    optimizer = tf.keras.optimizers.Adam()

    # prepare checkpoint checkpoint
    ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.ckpt_path, max_to_keep=1)

    # load checkpoint
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Restored from checkpoint: {}".format(ckpt_manager.latest_checkpoint))

    while True:
        img_name = input("Please specify image location: ")
        print(predict(FLAGS, encoder, decoder, tokenizer_wrapper, load_image(img_name)))
