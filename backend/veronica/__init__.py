import warnings

# NumPy 2.x linked against Apple Accelerate raises spurious FP warnings from matmul on
# Apple Silicon even for finite inputs/outputs; outputs are validated in tests.
warnings.filterwarnings("ignore", message=r".*encountered in matmul", category=RuntimeWarning)
# Depth Anything 3 registers OmegaConf resolvers with a deprecated call at import time.
warnings.filterwarnings("ignore", message=r"(?s).*register_resolver.*")
