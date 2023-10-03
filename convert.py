import argparse
import os
import shutil

import torch


def merge_weights(input_path, output_path, num_layers=60):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    weights = []
    for i in range(4):
        file_path = os.path.join(input_path, f"consolidated.0{i}.pth")
        weights.append(torch.load(file_path, map_location="cpu"))

    for weight in weights:
        for i in range(num_layers):
            prefix = f"layers.{i}.attention."
            wq = weight[prefix + "wq.weight"]
            wk = weight[prefix + "wk.weight"]

            wattn = torch.cat([wq, wk], dim=0)
            weight[prefix + "wqk.weight"] = wattn.cpu()

            del weight[prefix + "wq.weight"]
            del weight[prefix + "wk.weight"]

    tok_emb_total = torch.cat([w["tok_embeddings.weight"] for w in weights], dim=-1)

    for weight in weights:
        weight["tok_embeddings.weight"] = tok_emb_total

    for i, weight in enumerate(weights):
        file_path = os.path.join(output_path, f"consolidated.0{i}.pth")
        torch.save(weight, file_path)

    shutil.copy2(
        os.path.join(input_path, "params.json"),
        os.path.join(output_path, "params.json"),
    )


def main():
    parser = argparse.ArgumentParser(description="Merge weights of model files.")
    parser.add_argument(
        "input_path", type=str, help="Path to the original weight files."
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path where the merged weight files will be saved.",
    )

    args = parser.parse_args()

    merge_weights(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
