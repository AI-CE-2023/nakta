import argparse
import json
import os
import torch
import shutil

def read_json(path):
    with open(path, "r") as f:
        return json.loads(f.read())

def write_model(input_base_path, output_dir, filename_01, filename_23):
    params = read_json(os.path.join(input_base_path, "params.json"))
    n_layers = params["n_layers"]
    n_heads = params["n_heads"]
    num_shards = 4  # Since we are only considering the 30B case
    n_heads_per_shard = n_heads // num_shards
    dim = params["dim"]
    dims_per_head = dim // n_heads

    # Load weights
    loaded = [
        torch.load(os.path.join(input_base_path, f"consolidated.{i:02d}.pth"), map_location="cpu")
        for i in range(num_shards)
    ]

    for shard_group, filename in [((0,1), filename_01), ((2,3), filename_23)]:
        state_dict = {}
        for layer_i in range(n_layers):
            state_dict |= {
                f"layers.{layer_i}.attention_norm.weight": loaded[0][f"layers.{layer_i}.attention_norm.weight"],
                f"layers.{layer_i}.ffn_norm.weight": loaded[0][f"layers.{layer_i}.ffn_norm.weight"],
            }

            # Load attention weights based on shard_group and combine wq and wk into wqk
            wq = torch.cat(
                [
                    loaded[i][f"layers.{layer_i}.attention.wq.weight"].view(n_heads_per_shard, dims_per_head, dim)
                    for i in shard_group
                ],
                dim=0,
            ).reshape(dim//2, dim)
            wk = torch.cat(
                [
                    loaded[i][f"layers.{layer_i}.attention.wk.weight"].view(n_heads_per_shard, dims_per_head, dim)
                    for i in shard_group
                ],
                dim=0,
            ).reshape(dim//2, dim)
            wqk = torch.cat([wq, wk], dim=0).reshape(dim, dim)
            state_dict[f"layers.{layer_i}.attention.wqk.weight"] = wqk

            state_dict[f"layers.{layer_i}.attention.wv.weight"] = torch.cat(
                [
                    loaded[i][f"layers.{layer_i}.attention.wv.weight"].view(n_heads_per_shard, dims_per_head, dim)
                    for i in shard_group
                ],
                dim=0,
            ).reshape(dim//2, dim)
            state_dict[f"layers.{layer_i}.attention.wo.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.attention.wo.weight"] for i in shard_group], dim=1
            )
            state_dict[f"layers.{layer_i}.feed_forward.w1.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.feed_forward.w1.weight"] for i in shard_group], dim=0
            )
            state_dict[f"layers.{layer_i}.feed_forward.w2.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.feed_forward.w2.weight"] for i in shard_group], dim=1
            )
            state_dict[f"layers.{layer_i}.feed_forward.w3.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.feed_forward.w3.weight"] for i in shard_group], dim=0
            )

        state_dict |= {
            "norm.weight": loaded[0]["norm.weight"],
            "output.weight": torch.cat([loaded[i]["output.weight"] for i in shard_group], dim=0),
        }

        torch.save(state_dict, os.path.join(output_dir, filename))
    
    embedder= torch.cat(
        [loaded[i]["tok_embeddings.weight"] for i in range(num_shards)], dim=1
    )
    torch.save(embedder, os.path.join(output_dir, "embedder.pt"))
    shutil.copy2(
        os.path.join(input_base_path, "params.json"),
        os.path.join(output_dir, "params.json"),
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="./weights/original/30B",
        help="Location of LLaMA weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--output_dir",
        default="./weights/round2/30B",
        help="Directory where the merged weights will be saved",
    )
    parser.add_argument(
        "--filename_01",
        default="consolidated.01.pth",
        help="Filename for merged weights of shards 0 and 1",
    )
    parser.add_argument(
        "--filename_23",
        default="consolidated.02.pth",
        help="Filename for merged weights of shards 2 and 3",
    )
    args = parser.parse_args()

    write_model(
        input_base_path=args.input_dir,
        output_dir=args.output_dir,
        filename_01=args.filename_01,
        filename_23=args.filename_23
    )

if __name__ == "__main__":
    main()
