from .contact_angle_analyzer import (
    SlicedContactAngleAnalyzer,
    BinnedContactAngleAnalyzer,
    BaseContactAngleAnalyzer
)

def create_contact_angle_analyzer(method: str, parser, output_dir: str, **kwargs) -> BaseContactAngleAnalyzer:
    if method == "sliced":
        return SlicedContactAngleAnalyzer(parser=parser, output_repo=output_dir, **kwargs)
    elif method == "binned":
        return BinnedContactAngleAnalyzer(parser=parser, output_dir=output_dir, **kwargs)
    else:
        raise ValueError(f"Unknown method '{method}'. Expected 'sliced' or 'binned'.")
