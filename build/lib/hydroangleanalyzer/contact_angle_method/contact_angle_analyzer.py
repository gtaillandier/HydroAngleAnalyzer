from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from .binning_method.angle_fitting_binning import ContactAngle_binning
from .sliced_method.multi_processing import ContactAngle_sliced_parallel


class BaseContactAngleAnalyzer(ABC):
    """Abstract base for contact angle analysis across trajectories."""

    @abstractmethod
    def analyze(
        self, frame_range: Optional[List[int]] = None, **kwargs
    ) -> Dict[str, Any]:
        """Run the analysis and return statistics."""
        pass

    @abstractmethod
    def get_method_name(self) -> str:
        pass

    def summary(self) -> Dict[str, float]:
        """Return quick summary statistics."""
        results = self.analyze()
        return {
            "mean": results["mean_angle"],
            "std": results["std_angle"],
            "n_samples": len(results["angles"]),
        }


class SlicedContactAngleAnalyzer(BaseContactAngleAnalyzer):
    def __init__(self, parser, output_repo: str, **kwargs):
        self.parser = parser
        self.output_repo = output_repo
        self._processor = ContactAngle_sliced_parallel(
            filename=parser.in_path, output_repo=output_repo, **kwargs
        )

    def analyze(
        self, frame_range: Optional[List[int]] = None, **kwargs
    ) -> Dict[str, Any]:
        if frame_range is None:
            frame_range = list(range(self.parser.frame_tot()))

        frame_to_angle = self._processor.process_frames_parallel(
            frames_to_process=frame_range, **kwargs
        )
        angles = np.array(list(frame_to_angle.values()))

        return {
            "mean_angle": np.mean(angles),
            "std_angle": np.std(angles),
            "angles": frame_to_angle,
            "frames_analyzed": list(frame_to_angle.keys()),
            "method_metadata": {"frames_per_angle": 1},
        }

    def get_method_name(self) -> str:
        return "sliced_parallel"


class BinnedContactAngleAnalyzer(BaseContactAngleAnalyzer):
    def __init__(self, parser, output_dir: str, **kwargs):
        self.parser = parser
        self.output_dir = output_dir
        self._analyzer = ContactAngle_binning(
            parser=parser, output_dir=output_dir, **kwargs
        )

    def analyze(
        self,
        frame_range: Optional[List[int]] = None,
        split_factor: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if frame_range is None:
            frame_range = list(range(self.parser.frame_tot()))
        if split_factor is None:
            angle, _ = self._analyzer.process_batch(frame_range)
            angles = np.array([angle])
            method_metadata = {"frames_per_angle": len(frame_range)}
        else:
            angles = []
            for batch_idx, start in enumerate(range(0, len(frame_range), split_factor)):
                end = min(start + split_factor, len(frame_range))
                angle, _ = self._analyzer.process_batch(
                    frame_range[start:end],
                    batch_index=batch_idx + 1,  # Pass batch index
                )
                angles.append(angle)
            angles = np.array(angles)
            method_metadata = {"frames_per_trajectory": split_factor}
        return {
            "mean_angle": np.mean(angles),
            "std_angle": np.std(angles),
            "angles": angles,
            "frames_analyzed": frame_range,
            "method_metadata": method_metadata,
        }

    def get_method_name(self) -> str:
        return "binned_density"
