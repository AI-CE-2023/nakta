import os
import re
import torch
from tqdm.cli import tqdm

path_30b = './weights/original/30B/'

weights = {
  int(fn.split('.')[1]): torch.load(f'{path_30b}{fn}', map_location=torch.device('cpu'))
  for fn in tqdm(sorted(os.listdir(path_30b)))
  if fn.endswith('.pth')
}

# These tensors are duplicated rather than distributed among the files

not_distributed = {
  k 
  for k in weights[0].keys()
  if all((weights[0][k] == weights[i][k]).min() for i in range(1,4))
}

# What tensor dimensions should be merged, based on whether they are implemented
# as Embedding, Row or Column Parallel.

merge_dimensions ={
  r'^layers.\d+.attention.wq.weight$': 0,
  r'^layers.\d+.attention.wk.weight$': 0,
  r'^layers.\d+.attention.wv.weight$': 0,
  r'^layers.\d+.attention.wo.weight$': 1,

  r'^tok_embeddings.weight$': 1,

  r'^layers.\d+.feed_forward.w1.weight$': 0,
  r'^layers.\d+.feed_forward.w2.weight$': 1,
  r'^layers.\d+.feed_forward.w3.weight$': 0,
  r'^output.weight$': 0 
}

# Which files are merged into one
merge_groups = [[0,1], [2,3]]

# Merging (or copying if not distributed)
output_weights = {}
for output, group in enumerate(merge_groups):
  output_weights[output] = dict()
  for name in tqdm(weights[group[0]], leave=False):
    if name in not_distributed:
      output_weights[output][name] = weights[0][name]
    else:
      axis = next(axis for exp, axis in merge_dimensions.items() if re.match(exp, name))
      output_weights[output][name] = torch.cat([
          weights[member][name]
          for member in group
      ], axis=axis)

os.makedirs(f'{path_30b}/two-nodes/', exist_ok=True)
with open(f'{path_30b}/params.json') as fin:
  with open(f'{path_30b}/two-nodes/params.json', 'w') as fout:
    fout.write(fin.read())

torch.save(
    output_weights[0],
    f'{path_30b}/two-nodes/consolidated.00.pth'
)
torch.save(
    output_weights[1],
    f'{path_30b}/two-nodes/consolidated.01.pth'
)