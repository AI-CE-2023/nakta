<img src="./front.png" alt="front" width="896" height="512"/>

## Report
[Nakta Report](https://docs.google.com/document/d/12GCXtvHYw39m9fDLZdi5omF9eA2fgiUp4G7XDLnLyVA/edit?usp=sharing)
## 주요 성과
### Speed Test
#### Kernel Speed Test
![Kernel Speed](./kernels.png)

| Operation          | Performance Improvement (%) |
|--------------------|----------------------------|
| Attention          | 279.11%                    |
| RMSNorm            | 392.75%                    |
| RotaryEmbedding    | 133.37%                    |
| SwiGLU             | 33.53%                     |

#### Hellaswag Validation Set Speed Test
![Validation Speed](./speed.png)  
<br/>
Nakta vs LLAMA: 1.86x faster / Nakta with Cache vs LLAMA: 2.37x faster
### Accuracy Test
| Model  | Accuracy | Accuracy StdErr |
|--------|---------|-----------------|
| LLAMA  | 82.63%  | 0.38%           |
| Nakta  | 82.63%  | 0.38%           |

**Accuracy drop 없이 2.37배 빠른 모델 구현**
## 실행 가이드
### 환경 구축 가이드 

pytorch 2.0.1 CUDA 11.7, install [LLAMA Original Requirements](https://github.com/facebookresearch/llama/blob/llama_v1/requirements.txt), matplotlib, pandas  

1. **lm-evaluation-harness 리포지토리 클론 및 설치**
   - lm-evaluation-harness 리포지토리를 클론하고 설치합니다.
     ```bash
     git clone https://github.com/EleutherAI/lm-evaluation-harness 
     cd lm-evaluation-harness 
     pip install -e . 
     cd ..
     pip uninstall lm-eval
     ```

2. **Custom Cuda Kernel 리포지토리 클론 및 설치**
   - Custom Cuda Kernel 리포지토리를 클론하고 설치합니다.
     ```bash
     git clone https://github.com/AI-CE-2023/flash.git 
     cd flash 
     make install
     ```
     
3. **Triton 재설치**
   - 기존에 설치되어 있는 Triton을 제거하고, Triton-nightly를 설치합니다.
     ```bash
     pip uninstall -y triton 
     pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly==2.1.0.dev20230822000928
     ```  
### 설치 및 Weight 변환
```bash
git clone https://github.com/AI-CE-2023/nakta.git 
cd nakta 
python convert.py {Your Original Weight Path} {Your Output Path}
```  
 다음 weigth 변환은 weight 내용을 변환하는 것이 아닌 Rotary Embedding 시에 Query, Key 를 한번에 넣어주기 위해 Weight 를 합치는 내용입니다. 또한 Parallel Embedding 을 Normal Embedding 으로 바꾸기 위해 합친 Weight 에 대한 내용을 담고 있습니다.  
```
cd speed_bench
torchrun --nproc_per_node 4 nakta_speed.py 
torchrun --nproc_per_node 4 llama_speed.py
```
*llama_speed 와 nakta_speed 를 실행하기전 weigth 와 tokenizer 의 경로를 수정 부탁드립니다.
## 사용 환경
CUDA 11.7 Torch 2.0.1 triton-nightly 2.1.0.dev20230822000928 
<br/>
dockerfile 의 경우 docker 설치가 안 되는 작업 환경에서 작업하여 작동을 장담할 수 없습니다. 설치 내용은 같으니 참고 부탁드립니다.
## Repository 가이드  
accuracy_test: hellaswag validation set 에 대한 accuracy test  
kernel_benchmark: kernel benchmark + graph  
llama_org: LLAMA-1 original implementation  
model_profile:  nsys profile results of LLAMA-1, Nakta, Nakta with cache  
nakta_model: Nakta Model implementation  
speed_bench: speed benchmark with hellaswag validation set  

## Kernel 구현부 링크
[RMSNorm](https://github.com/AI-CE-2023/nakta/blob/main/nakta_model/kernel/Norm/RmsNorm.py)   
[Rotary Embedding](https://github.com/AI-CE-2023/nakta/blob/main/nakta_model/kernel/Emb/Rotary/rotary.py)   
[SwiGLU](https://github.com/AI-CE-2023/flash/blob/main/csrc/flash_attn/activation_kernel.cu)  
*Memory Efficient Attention 의 경우 Pytorch 2.0.1 의 구현을 사용하였음.

## 참고 프로젝트 
  - Accuracy Test: https://github.com/EleutherAI/lm-evaluation-harness
  - Rotary Embedding, RMSNorm: https://github.com/openai/triton https://github.com/Dao-AILab/flash-attention
  
