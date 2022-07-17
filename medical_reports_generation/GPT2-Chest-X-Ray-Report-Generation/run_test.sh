cd /mnt/proj2/open-24-11/gpt2-attention_oscar_fulldata/GPT2-Chest-X-Ray-Report-Generation
ml cuDNN/8.2.2.26-CUDA-11.4.1
echo $CUDA_VISIBLE_DEVICES

python3 test.py
