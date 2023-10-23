import functools

import torch


def nvtx_annotate(module_cls):
    """
    Decorator to wrap forward method of an nn.Module to automatically push and pop NVTX ranges.
    """
    # Store original forward method
    original_forward = module_cls.forward

    @functools.wraps(original_forward)
    def new_forward(self, *args, **kwargs):
        # Push NVTX range with the name of the module class
        torch.cuda.nvtx.range_push(f"{module_cls.__name__}")
        
        # Call the original forward method
        result = original_forward(self, *args, **kwargs)
        
        # Pop NVTX range
        torch.cuda.nvtx.range_pop()
        
        return result

    # Replace the forward method with the new one
    module_cls.forward = new_forward

    return module_cls

def nvtx_annotate_function(func):
    """
    Decorator to wrap a function to automatically push and pop NVTX ranges.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Push NVTX range with the name of the function
        torch.cuda.nvtx.range_push(f"{func.__name__}")
        
        # Call the original function
        result = func(*args, **kwargs)
        
        # Pop NVTX range
        torch.cuda.nvtx.range_pop()
        
        return result

    return wrapper
