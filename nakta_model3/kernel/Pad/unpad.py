import torch


def remove_padding(output, batch_info_set):
    # Flatten the sequences based on batch_info
    if type(batch_info_set) == int:
        return output
    flattened_output = torch.cat(
        [output[i, : batch_info_set[0][i]] for i in range(len(batch_info_set[0]))],
        dim=0,
    )
    return flattened_output
