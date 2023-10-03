# nakta

## 주요 성과
### Speed Test
#### Kernel Speed Test
![Kernel Speed]('./kernel.png')

| Operation          | Performance Improvement (%) |
|--------------------|----------------------------|
| Attention          | 279.11%                    |
| RMSNorm            | 392.75%                    |
| RotaryEmbedding    | 133.37%                    |
| SwiGLU             | 33.53%                     |

#### Hellaswag Validation Set Speed Test
![Validation Speed]('./speed.png')
Nakta vs LLAMA: 1.86x faster / Nakta with Cache vs LLAMA: 2.37x faster
### Accuracy Test
| Model  | Accuracy | Accuracy StdErr |
|--------|---------|-----------------|
| LLAMA  | 82.63%  | 0.38%           |
| Nakta  | 82.63%  | 0.38%           |
