# MITR
Source code of paper: Active-Inference-Guided Bayesian Memory Evolution for Continual Learning Reasoning Model



This repository contains the code to reproduce the `relational reasoning: sort_of_clever`,`text-based question-answering: bAbI`, `detecting equilateral triangles` and `cifar-10` tasks from our paper.  


## Install relevant libraries
```
pip install -r requirements.txt 
```
## Task1: Sort-of-CLEVR
You can find the source code for the Sort-of-CLEVR task in `sort_of_clevr_and_babi` folder.

Firstly, dataset generation:
```
python sort_of_clevr_generator.py
```
**Execute the following commands to reproduce all experiments for the Triangle task in the paper:**
```
sh sort.sh h_dim num_layers num_heads share_vanilla_parameters use_topk topk shared_memory_attention mem_slots use_long_men long_mem_segs long_mem_aggre use_wm_inference seed set_transformer
```
**Explanation of Parameters:**

`h_dim`: Embedding dimensions

`num_layers`: Number of model layers

`num_heads`: Number of heads in multi-headed attention

`share_vanilla_parameters`: Whether share parameters across layers.

`use_topk`: Whether to use top-k competition

`topk`: Value of k in top-k competition

`shared_memory_attention`: Whether to use shared working memory and long-term memory. 
 If shared_memory_attention is false, then vanilla multi-head attention is used.

`mem_slots`: Number of slots in working memory

`use_long_men`: Whether to use long-term memory component. 

`long_mem_segs`: Number of long-term memory segments

`long_mem_aggre`: Whether cross-attention is performed on information retrieved from the working memory and long-term memory.

`use_wm_inference`: Whether working memory come into play during the reasoning process

`seed`: Random seed

`functional`: ues Set Transformer or not.

**Specifically, please execute the following commands to reproduce all experiments for the Sort-of-CLEVR task in the paper:**

```
MITR 
sh sort.sh 256 8 8 True True 5 True 7 True 5 True True 1 False

HSWTR
sh sort.sh 256 8 8 True True 5 True 7 False 5 False False 1 False

SDMTR
sh sort.sh 256 8 8 True False 5 False 7 False 5 False False 1 False

TR
sh sort.sh 256 4 4 False False 5 False 7 False 5 False False 1 False

STR
sh sort.sh 256 4 4 False False 5 False 7 False 5 False False 1 True
```

## Task2: bAbI
You can find the source code for the bAbI task in `sort_of_clevr_and_babi` folder.

**Execute the following commands to reproduce experiment for the bAbI task in the paper:**
```
sh babi.sh h_dim num_layers num_heads share_vanilla_parameters use_topk topk shared_memory_attention mem_slots use_long_men long_mem_segs long_mem_aggre use_wm_inference seed set_transformer
```

```
MITR
sh babi.sh 256 8 8 True True 5 True 7 True 5 True True 1 False
```

## Task3: Detecting Equilateral Triangles 
You can find the source code for the Triangle task in `triangle_and_cifar10` folder.

**Execute the following commands to reproduce all experiments for the Triangle task in the paper:**

```
sh run.sh dataset model patch_size num_layers h_dim ffn_dim share_vanilla_parameters use_topk topk
shared_memory_attention mem_slots use_long_men long_mem_segs long_mem_aggre use_wm_inference seed
```

```
MITR
sh run.sh "Triangle" "default" 32 2 128 256 True True 5 True 7 True 5 True True 1

HSWTR
sh run.sh "Triangle" "default" 4 4 128 256 True True 5 True 7 False 5 False True 1

SDMTR
sh run.sh "Triangle" "default" 4 4 128 256 True False 5 False 7 False 5 False True 1

TR
sh run.sh "Triangle" "default" 4 4 128 256 True False 5 False 7 False 5 False True 1

STR
sh run.sh "Triangle" "functional" 4 4 128 256 False False 5 False 7 False 5 False True 1
```

## Task4: Image Classification
You can find the source code for the Cifar-10 task in `triangle_and_cifar10` folder.

**Execute the following commands to reproduce all experiments for the cifar10 task in the paper:**

```
sh run.sh dataset model patch_size num_layers h_dim ffn_dim share_vanilla_parameters use_topk topk
shared_memory_attention mem_slots use_long_men long_mem_segs long_mem_aggre use_wm_inference seed
```

```
MITR
sh run.sh "cifar10" "default" 4 4 256 256 True True 5 True 8 True 5 True True 1

HSWTR
sh run.sh "cifar10" "default" 4 4 256 256 True True 5 True 8 False 5 False True 1

SDMTR
sh run.sh "Triangle" "default" 4 4 128 256 True False 5 False 7 False 5 False True 1

TR
sh run.sh "cifar10" "default" 4 4 256 256 True False 5 False 8 False 5 False True 1

STR
sh run.sh "cifar10" "functional" 4 4 256 256 False False 5 False 8 False 5 False True 1
```

## Task5: Text Generation
You can find the source code for the Text8 task in `sort_of_clevr_and_babi` folder.

**Execute the following command to reproduce experiment for the Text8 task in the paper:**

```
python llm_main.py --data ./data/text8 --dataset text8 --num_layers 4 --d_embed 512 --d_model 512 --embed_dim 512 --seed 1 --log-interval 1000 --eval-interval 5000 --batch_size 16 --tgt_len 70 --eval_tgt_len 50 --max_step 100000 --dropout 0.1 --cuda
```
