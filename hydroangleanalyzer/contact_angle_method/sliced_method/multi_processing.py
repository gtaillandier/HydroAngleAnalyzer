import os
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Tuple, Type
import numpy as np
import logging
from hydroangleanalyzer.contact_angle_method.sliced_method import ContactAngle_sliced
from hydroangleanalyzer.parser import DumpParser, Ase_Parser, XYZ_Parser, BaseParser
from hydroangleanalyzer.io_utils import detect_parser_type
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContactAngle_sliced_parallel():
    """
    A parallel frame processor that uses batch processing to avoid OVITO pickle issues.
    """

    def __init__(
                self,
                filename: str,
                output_repo: str,
                max_dist: float = 100,
                wall_max_z: float = 4.8,
                type_model: str = 'spherical',
                particle_type_wall: set = {1},
                liquid_indices: np.ndarray = np.array([]),
                delta_gamma: float = None,
                delta_masspain: float = None,
            ):
        """
        Initialize the BatchFrameProcessor.

        Args:
            filename: Path to the dump file.
            output_repo: Output directory for results.
            delta_gamma: Gamma parameter for contact angle calculation.
            max_dist: Maximum distance for analysis.
            wall_max_z: Maximum Z coordinate for wall.
            delta_masspain: Y-axis delta parameter.
            type_model: Type of analysis ('spherical' or other).
            particle_type_wall: Set of particle types for wall.
            liquid_indices: array of indices.
        """
        self.filename = filename
        self.output_repo = output_repo
        self.delta_gamma = delta_gamma
        self.max_dist = max_dist
        self.wall_max_z = wall_max_z
        self.delta_masspain = delta_masspain
        self.type_model = type_model
        self.particle_type_wall = particle_type_wall
        self.liquid_indices = liquid_indices
        # Removed undefined variable: self.particule_liquid_type

        # Ensure output directory exists
        os.makedirs(self.output_repo, exist_ok=True)

    def process_frames_parallel(self, frames_to_process: List[int],
                                num_batches: int = 4,
                                max_workers: Optional[int] = None) -> Dict[int, float]:
        """
        Process multiple frames in parallel using batch processing.

        Args:
            frames_to_process: List of frame numbers to process.
            num_batches: Number of batches to create.
            max_workers: Maximum worker processes.

        Returns:
            Dictionary mapping frame numbers to mean contact angles.
        """
        if max_workers is None:
            max_workers = num_batches
        
        # Create batches
        batches = self._create_batches(frames_to_process, num_batches)
        logger.info(f"Processing {len(frames_to_process)} frames in {len(batches)} batches with {max_workers} workers")

        # Process batches in parallel
        results = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self._process_batch_worker, batch_frames): batch_frames
                for batch_frames in batches
            }

            # Collect results as they complete
            completed_batches = 0
            for future in as_completed(future_to_batch):
                batch_frames = future_to_batch[future]
                try:
                    batch_results = future.result()
                    completed_batches += 1
                    logger.info(f"Completed batch {completed_batches}/{len(batches)} ({len(batch_results)} frames)")
                    
                    # Add results to main dictionary
                    for frame_num, mean_alpha in batch_results:
                        if mean_alpha is not None:
                            results[frame_num] = mean_alpha
                except Exception as e:
                    logger.error(f"Error in batch for frames {batch_frames}: {e}", exc_info=True)

        logger.info(f"Successfully processed {len(results)}/{len(frames_to_process)} frames")
        return results

    def _create_batches(self, frames: List[int], num_batches: int) -> List[List[int]]:
        """
        Split frames into approximately equal batches.

        Args:
            frames: List of frame numbers.
            num_batches: Number of batches to create.

        Returns:
            List of batches.
        """
        if num_batches >= len(frames):
            return [[frame] for frame in frames]
        
        batch_size = math.ceil(len(frames) / num_batches)
        batches = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            batches.append(batch)
        return batches

    def _process_batch_worker(self, batch_frames: List[int]) -> List[Tuple[int, Optional[float]]]:
        """
        Worker function that processes a batch of frames in a single process.

        Args:
            batch_frames: List of frame numbers in this batch.

        Returns:
            Results for each frame in the batch.
        """
        try:
            from hydroangleanalyzer.parser.parser_dump import Dump_WaterMoleculeFinder, DumpParser
            from hydroangleanalyzer.parser.parser_ase import Ase_Parser
            from hydroangleanalyzer.parser.parser_xyz import XYZ_Parser
            from hydroangleanalyzer.parser.base_parser import BaseParser
            from hydroangleanalyzer.contact_angle_method.sliced_method.angle_fitting_sliced import ContactAngle_sliced
            from hydroangleanalyzer.io_utils import detect_parser_type
                
        except ImportError as e:
            logger.error(f"Failed to import required classes: {e}")
            return [(frame, None) for frame in batch_frames]

        try:
            # Determine the parser class to use based on file extension
            parser_type = detect_parser_type(self.filename)
            logger.info(f"Detected parser type: {parser_type}")
            parser_class: Type[BaseParser]
            if parser_type == 'dump':
                parser_class = DumpParser
            elif parser_type == 'ase':
                parser_class = Ase_Parser
            elif parser_type == 'xyz':
                parser_class = XYZ_Parser
            else:
                raise ValueError(f"Unsupported parser type: {parser_type}")

            # Initialize the parser once per batch
            parser = parser_class(
                in_path=self.filename,
                particle_type_wall=self.particle_type_wall,
            )

        except Exception as e:
            logger.error(f"Error initializing parser: {e}")
            return [(frame, None) for frame in batch_frames]

        batch_results = []
        for frame_num in batch_frames:
            try:
                # Process the frame - use self.liquid_indices
                result = self._process_single_frame_with_parsers(frame_num, self.liquid_indices, parser)
                batch_results.append(result)
            except Exception as e:
                logger.error(f"Error processing frame {frame_num}: {e}")
                batch_results.append((frame_num, None))
        return batch_results

    def _process_single_frame_with_parsers(self, frame_num: int,
                                           liquid_indices: np.ndarray,  
                                           parser: BaseParser) -> Tuple[int, Optional[float]]:
        """
        Process a single frame using provided parser instances.

        Args:
            frame_num: Frame number to process.
            liquid_indices: Array of liquid particle indices.
            parser: BaseParser instance.

        Returns:
            Frame number and mean contact angle.
        """
        logger.info(f"START processing frame {frame_num}")
        try:
            # Parse positions of liquid particles
            liquid_positions = parser.parse(num_frame=frame_num, indices=liquid_indices)

            max_dist = int(np.max(np.array([parser.box_size_y(num_frame=frame_num), parser.box_size_x(num_frame=frame_num)])) / 2)
            logger.info(f"Frame {frame_num}: Parsed {len(liquid_positions)} liquid particles with max_dist {max_dist}")
            if self.type_model == 'masspain_x':
                liquid_positions = liquid_positions[:, [1, 0, 2]]
            else:
                liquid_positions = liquid_positions
            # Get box dimensions for the frame
            if self.type_model == 'masspain_x':
                box_dimensions = parser.box_size_x(num_frame=frame_num)
            elif self.type_model == 'masspain_y':
                box_dimensions = parser.box_size_y(num_frame=frame_num)
            else:
                box_dimensions = None
            # Calculate mean position of liquid particles (NOT from indices!)
            mean_liquid_position = np.mean(liquid_positions, axis=0)
            # Predict contact angle
            predictor = ContactAngle_sliced(
                o_coords=liquid_positions,
                max_dist=max_dist,
                o_center_geom=mean_liquid_position,
                type_model=self.type_model,
                delta_gamma=self.delta_gamma,
                width_masspain=box_dimensions,
                delta_masspain=self.delta_masspain
            )

            list_1, list_2, list_3 = predictor.predict_contact_angle()
            mean_alpha = np.mean(list_1)

            # Save results
            np.savetxt(f"{self.output_repo}/alfasframe{frame_num}.txt", np.array(list_1), fmt='%f')
            np.save(f"{self.output_repo}/surfacesframe{frame_num}.npy", np.array(list_2))
            np.save(f"{self.output_repo}/poptsframe{frame_num}.npy", np.array(list_3))

            logger.info(f"Frame {frame_num} - mean angle: {mean_alpha:.2f}°")
            return frame_num, mean_alpha

        except Exception as e:
            logger.error(f"Error processing frame {frame_num}: {e}")
            return frame_num, None


