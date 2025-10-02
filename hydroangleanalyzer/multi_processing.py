import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict, Optional, Set, Any
import math
from pathlib import Path

from hydroangleanalyzer import ContactAnglePredictor, BaseParser

class BatchFrameProcessor:
    """
    Parallel frame processor for contact angle analysis with batch processing.
    
    This class processes molecular dynamics frames in parallel by grouping them into batches,
    where each batch is processed by a single worker process. This approach optimizes
    performance while avoiding serialization issues with complex objects.
    
    The processor uses a parser-agnostic design, working with any BaseParser implementation
    (LAMMPS, XYZ, ASE, etc.) and requires pre-identified liquid particle indices.
    
    Attributes:
        in_path (str): Path to the trajectory file.
        output_repo (Path): Output directory for results.
        parser_class (type): Parser class to instantiate (must inherit from BaseParser).
        parser_kwargs (Dict[str, Any]): Keyword arguments for parser initialization.
        liquid_indices (np.ndarray): Indices of liquid particles to analyze.
        delta_gamma (float): Gamma parameter for contact angle calculation.
        max_dist (float): Maximum distance for analysis.
        delta_y_axis (float): Y-axis delta parameter.
        analysis_type (str): Type of analysis ('spherical' or other).
    
    Example:
        >>> from hydroangleanalyzer import DumpParser, WaterMoleculeFinder
        >>> 
        >>> # Step 1: Identify water oxygen atoms
        >>> finder = WaterMoleculeFinder(
        ...     in_path='trajectory.lammpstrj',
        ...     particle_type_wall={3},
        ...     oxygen_type=1,
        ...     hydrogen_type=2
        ... )
        >>> oxygen_indices = finder.get_water_oxygen_indices(num_frame=0)
        >>> 
        >>> # Step 2: Initialize batch processor
        >>> processor = BatchFrameProcessor(
        ...     in_path='trajectory.lammpstrj',
        ...     output_repo='results/',
        ...     parser_class=DumpParser,
        ...     parser_kwargs={'particle_type_wall': {3}},
        ...     liquid_indices=oxygen_indices,
        ...     delta_gamma=5.0,
        ...     max_dist=100.0
        ... )
        >>> 
        >>> # Step 3: Process frames in parallel
        >>> frames = list(range(0, 1000, 50))
        >>> results = processor.process_frames_parallel(frames, num_batches=4)
        >>> print(f"Mean contact angle: {np.mean(list(results.values())):.2f}°")
    """
    
    def __init__(
        self,
        in_path: str,
        output_repo: str,
        parser_class: type,
        parser_kwargs: Dict[str, Any],
        liquid_indices: np.ndarray,
        delta_gamma: float = 5.0,
        max_dist: float = 100.0,
        delta_y_axis: float = 1.0,
        analysis_type: str = 'spherical'
    ):
        """
        Initialize the BatchFrameProcessor.
        
        Args:
            in_path: Path to the trajectory file.
            output_repo: Output directory for results.
            parser_class: Parser class to use (must inherit from BaseParser).
            parser_kwargs: Keyword arguments for parser initialization.
            liquid_indices: Indices of liquid particles (e.g., water oxygen atoms).
            delta_gamma: Gamma parameter for contact angle calculation (default: 5.0).
            max_dist: Maximum distance for analysis (default: 100.0).
            delta_y_axis: Y-axis delta parameter (default: 1.0).
            analysis_type: Type of analysis, 'spherical' or other (default: 'spherical').
        
        Raises:
            ValueError: If parser_class doesn't inherit from BaseParser.
            FileNotFoundError: If in_path doesn't exist.
        """
        # Validate inputs
        if not issubclass(parser_class, BaseParser):
            raise ValueError(
                f"parser_class must inherit from BaseParser, got {parser_class.__name__}"
            )
        
        if not os.path.exists(in_path):
            raise FileNotFoundError(f"Trajectory file not found: {in_path}")
        
        if len(liquid_indices) == 0:
            raise ValueError("liquid_indices cannot be empty")
        
        # Store configuration
        self.in_path = in_path
        self.output_repo = Path(output_repo)
        self.parser_class = parser_class
        self.parser_kwargs = parser_kwargs
        self.liquid_indices = np.asarray(liquid_indices, dtype=int)
        
        # Analysis parameters
        self.delta_gamma = delta_gamma
        self.max_dist = max_dist
        self.delta_y_axis = delta_y_axis
        self.analysis_type = analysis_type
        
        # Create output directory
        self.output_repo.mkdir(parents=True, exist_ok=True)
        
        print(f"BatchFrameProcessor initialized:")
        print(f"  - Trajectory: {self.in_path}")
        print(f"  - Parser: {parser_class.__name__}")
        print(f"  - Liquid particles: {len(self.liquid_indices)}")
        print(f"  - Output: {self.output_repo}")
    
    def process_frames_parallel(
        self,
        frames_to_process: List[int],
        num_batches: int = 4,
        max_workers: Optional[int] = None
    ) -> Dict[int, float]:
        """
        Process multiple frames in parallel using batch processing.
        
        This method divides frames into batches and processes each batch in a separate
        worker process, optimizing parallel efficiency while avoiding serialization issues.
        
        Args:
            frames_to_process: List of frame numbers to process.
            num_batches: Number of batches to create (default: 4).
            max_workers: Maximum worker processes (default: num_batches).
        
        Returns:
            Dictionary mapping frame numbers to mean contact angles.
        
        Example:
            >>> frames = [0, 50, 100, 150, 200]
            >>> results = processor.process_frames_parallel(frames, num_batches=2)
            >>> print(f"Processed {len(results)} frames")
            Processed 5 frames
        """
        if max_workers is None:
            max_workers = num_batches
        
        # Create batches
        batches = self._create_batches(frames_to_process, num_batches)
        print(f"\nProcessing {len(frames_to_process)} frames in {len(batches)} batches "
              f"with {max_workers} workers")
        
        # Process batches in parallel
        results = {}
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches
            future_to_batch = {}
            for i, batch_frames in enumerate(batches):
                future = executor.submit(
                    self._process_batch_worker,
                    batch_frames,
                    self.in_path,
                    self.parser_class,
                    self.parser_kwargs,
                    self.liquid_indices,
                    self.delta_gamma,
                    self.max_dist,
                    self.delta_y_axis,
                    self.analysis_type,
                    str(self.output_repo)
                )
                future_to_batch[future] = (i, len(batch_frames))
            
            # Collect results as they complete
            completed_batches = 0
            total_processed = 0
            
            for future in as_completed(future_to_batch):
                batch_idx, batch_size = future_to_batch[future]
                try:
                    batch_results = future.result()
                    completed_batches += 1
                    total_processed += len(batch_results)
                    
                    # Calculate success rate for this batch
                    successful = sum(1 for _, angle in batch_results if angle is not None)
                    success_rate = (successful / len(batch_results)) * 100
                    
                    print(f"  Batch {completed_batches}/{len(batches)}: "
                          f"{successful}/{len(batch_results)} frames successful "
                          f"({success_rate:.1f}%)")
                    
                    # Add results to main dictionary
                    for frame_num, mean_alpha in batch_results:
                        if mean_alpha is not None:
                            results[frame_num] = mean_alpha
                            
                except Exception as e:
                    print(f"  ERROR in batch {batch_idx + 1}: {e}")
        
        # Summary
        success_rate = (len(results) / len(frames_to_process)) * 100
        print(f"\nCompleted: {len(results)}/{len(frames_to_process)} frames "
              f"({success_rate:.1f}% success)")
        
        return results
    
    def process_frame_single(self, frame_num: int) -> Tuple[int, Optional[float]]:
        """
        Process a single frame (for testing or sequential processing).
        
        Args:
            frame_num: Frame number to process.
            
        Returns:
            Tuple of (frame_num, mean_contact_angle). mean_contact_angle is None on error.
        
        Example:
            >>> frame_num, mean_angle = processor.process_frame_single(0)
            >>> print(f"Frame {frame_num}: {mean_angle:.2f}°")
            Frame 0: 85.32°
        """
        # Initialize parser
        parser = self.parser_class(self.in_path, **self.parser_kwargs)
        
        return self._process_single_frame_with_parser(
            frame_num=frame_num,
            parser=parser,
            liquid_indices=self.liquid_indices,
            delta_gamma=self.delta_gamma,
            max_dist=self.max_dist,
            delta_y_axis=self.delta_y_axis,
            analysis_type=self.analysis_type,
            output_repo=str(self.output_repo)
        )
    
    def get_batch_info(
        self,
        frames_to_process: List[int],
        num_batches: int
    ) -> Dict[str, Any]:
        """
        Get information about how frames will be distributed across batches.
        
        Args:
            frames_to_process: List of frame numbers.
            num_batches: Number of batches.
            
        Returns:
            Dictionary with batch distribution information.
        
        Example:
            >>> info = processor.get_batch_info([0, 50, 100, 150, 200], 2)
            >>> print(f"Batch sizes: {info['batch_sizes']}")
            Batch sizes: [3, 2]
        """
        batches = self._create_batches(frames_to_process, num_batches)
        
        return {
            'total_frames': len(frames_to_process),
            'num_batches': len(batches),
            'batch_sizes': [len(batch) for batch in batches],
            'frames_per_batch': [batch for batch in batches],
            'max_batch_size': max(len(batch) for batch in batches),
            'min_batch_size': min(len(batch) for batch in batches),
            'avg_batch_size': len(frames_to_process) / len(batches)
        }
    
    @staticmethod
    def _create_batches(frames: List[int], num_batches: int) -> List[List[int]]:
        """
        Split frames into approximately equal batches.
        
        Args:
            frames: List of frame numbers.
            num_batches: Number of batches to create.
            
        Returns:
            List of batches (each batch is a list of frame numbers).
        """
        if num_batches >= len(frames):
            # One frame per batch if more batches than frames
            return [[frame] for frame in frames]
        
        batch_size = math.ceil(len(frames) / num_batches)
        batches = []
        
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            batches.append(batch)
        
        return batches
    
    @staticmethod
    def _process_batch_worker(
        batch_frames: List[int],
        in_path: str,
        parser_class: type,
        parser_kwargs: Dict[str, Any],
        liquid_indices: np.ndarray,
        delta_gamma: float,
        max_dist: float,
        delta_y_axis: float,
        analysis_type: str,
        output_repo: str
    ) -> List[Tuple[int, Optional[float]]]:
        """
        Worker function that processes a batch of frames in a single process.
        
        This static method is called by each worker process and initializes the parser
        once per batch to minimize overhead.
        
        Args:
            batch_frames: List of frame numbers in this batch.
            in_path: Path to trajectory file.
            parser_class: Parser class to instantiate.
            parser_kwargs: Keyword arguments for parser.
            liquid_indices: Indices of liquid particles.
            delta_gamma: Gamma parameter.
            max_dist: Maximum distance.
            delta_y_axis: Y-axis delta.
            analysis_type: Analysis type.
            output_repo: Output directory.
            
        Returns:
            List of tuples (frame_num, mean_angle) for each frame in batch.
        """
        # Initialize parser once per batch
        parser = parser_class(in_path, **parser_kwargs)
        
        batch_results = []
        
        for frame_num in batch_frames:
            try:
                result = BatchFrameProcessor._process_single_frame_with_parser(
                    frame_num=frame_num,
                    parser=parser,
                    liquid_indices=liquid_indices,
                    delta_gamma=delta_gamma,
                    max_dist=max_dist,
                    delta_y_axis=delta_y_axis,
                    analysis_type=analysis_type,
                    output_repo=output_repo
                )
                batch_results.append(result)
                
            except Exception as e:
                print(f"  ERROR processing frame {frame_num}: {e}")
                batch_results.append((frame_num, None))
        
        return batch_results
    
    @staticmethod
    def _process_single_frame_with_parser(
        frame_num: int,
        parser: BaseParser,
        liquid_indices: np.ndarray,
        delta_gamma: float,
        max_dist: float,
        delta_y_axis: float,
        analysis_type: str,
        output_repo: str
    ) -> Tuple[int, Optional[float]]:
        """
        Process a single frame using a provided parser instance.
        
        Args:
            frame_num: Frame number to process.
            parser: Parser instance.
            liquid_indices: Indices of liquid particles.
            delta_gamma: Gamma parameter.
            max_dist: Maximum distance.
            delta_y_axis: Y-axis delta.
            analysis_type: Analysis type.
            output_repo: Output directory.
            
        Returns:
            Tuple of (frame_num, mean_contact_angle).
        
        Raises:
            Exception: If frame processing fails.
        """
        # Parse liquid particle positions
        liquid_positions = parser.parse(num_frame=frame_num, indices=liquid_indices)
        
        if len(liquid_positions) == 0:
            raise ValueError(f"No liquid particles found in frame {frame_num}")
        
        # Calculate mean position (droplet center)
        mean_position = np.mean(liquid_positions, axis=0)
        
        # Create predictor and calculate contact angles
        predictor = ContactAnglePredictor(
            liquid_positions,
            delta_gamma,
            max_dist,
            mean_position,
            10,  # Fixed parameter (consider making configurable)
            delta_y_axis,
            type=analysis_type
        )
        
        contact_angles, surfaces, popts = predictor.predict_contact_angle()
        mean_alpha = np.mean(contact_angles)
        
        # Save results
        output_path = Path(output_repo)
        np.savetxt(
            output_path / f"alfasframe{frame_num}.txt",
            np.array(contact_angles),
            fmt='%.6f',
            header=f'Contact angles for frame {frame_num}'
        )
        np.save(
            output_path / f"surfacesframe{frame_num}.npy",
            np.array(surfaces)
        )
        np.save(
            output_path / f"poptsframe{frame_num}.npy",
            np.array(popts)
        )
        
        return frame_num, mean_alpha