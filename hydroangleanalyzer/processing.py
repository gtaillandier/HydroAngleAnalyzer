import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from hydroangleanalyzer import ContactAnglePredictor, DumpParser, DumpParse_wall
from typing import List, Tuple, Dict, Optional, Union


class FrameProcessor:
    def __init__(self, filename, output_repo, delta_gamma=5, max_dist=100, wall_max_z=4.8, delta_y_axis=1, type='spherical', particle_type_wall={2, 3},particule_liquid_type={1,2}, value_dist_marge_wall_liquide=5):
        self.filename = filename
        self.output_repo = output_repo
        self.delta_gamma = delta_gamma
        self.particule_liquid_type =particule_liquid_type
        self.max_dist = max_dist
        self.wall_max_z = wall_max_z
        self.delta_y_axis = delta_y_axis
        self.value_dist_marge_wall_liquide  = value_dist_marge_wall_liquide 
        self.type = type
        self.particle_type_wall = particle_type_wall
        # Ensure output directory exists
        os.makedirs(self.output_repo, exist_ok=True)

    def process_frame(self, frame_num ):
        """
        Process a single frame to calculate contact angles and save results.
        
        Args:
            frame_num (int): Frame number to process.
        
        Returns:
            tuple: Frame number and mean contact angle.
        """
        file_parser = DumpParser(self.filename, self.particle_type_wall)
        parsed_wall = DumpParse_wall(self.filename, self.particule_liquid_type)
        parsed_xyz = file_parser.parse(frame_num)
        highest__part_wall = self.parsed_wall.find_highest_wall_part(frame_num)
        mean_parsed = np.mean(parsed_xyz, axis=0)
        
        classpredictor = ContactAnglePredictor(
            parsed_xyz, self.delta_gamma, self.max_dist, mean_parsed,  highest__part_wall, 10, self.delta_y_axis,  limit_dist_wall= highest__part_wall + self.value_dist_marge_wall_liquide , type=self.type
        )
        
        list_1, list_2, list_3 = classpredictor.predict_contact_angle()
        mean_alpha = np.mean(list_1)
        
        np.savetxt(f"{self.output_repo}/alfasframe{frame_num}.txt", np.array(list_1), fmt='%f')
        np.save(f"{self.output_repo}/surfacesframe{frame_num}.npy", np.array(list_2))
        np.save(f"{self.output_repo}/poptsframe{frame_num}.npy", np.array(list_3))
        print(f"Frame {frame_num} - mean angle: {mean_alpha}")
        
        return frame_num, mean_alpha
    def process_frames_batch(self, frame_numbers, batch_size=100):
        """Process frames in batches to manage memory"""
        results = []
        
        for i in range(0, len(frame_numbers), batch_size):
            batch = frame_numbers[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}: frames {batch[0]}-{batch[-1]}")
            
            for frame_num in batch:
                result = self.process_frame(frame_num)
                results.append(result)
            
            # Force garbage collection between batches
            import gc
            gc.collect()
        
        return results
    def parallel_process_frames(self, frames, max_workers=None):
        """
        Process multiple frames in parallel using ProcessPoolExecutor.
        
        Args:
            frames (list): List of frame numbers to process.
            max_workers (int, optional): Maximum number of worker processes.
                                         Defaults to number of CPU cores.
        
        Saves:
            Combined mean contact angles to a text file.
        """
        if max_workers is None:
            max_workers = os.cpu_count()
            
        print(f'Using {max_workers} worker processes')
        results = []
        
        # Use ProcessPoolExecutor instead of multiprocessing.Pool
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks and store futures
            futures = {executor.submit(self.process_frame, frame): frame for frame in frames}
            
            # Collect results as they complete
            for future in futures:
                try:
                    frame_num, mean_alpha = future.result()
                    results.append([frame_num, mean_alpha])
                except Exception as e:
                    print(f"Error processing frame {futures[future]}: {e}")
        
        # Sort results by frame number
        results = sorted(results, key=lambda x: x[0])
        np.savetxt(f"{self.output_repo}/alfas_per_frame_combined.txt", np.array(results), fmt='%f')
        print("Saved all mean alphas to 'alfas_per_frame_combined.txt'")
        
        return results

    

import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from hydroangleanalyzer import ContactAnglePredictor, DumpParser, DumpParse_wall
from typing import List, Tuple, Dict, Optional, Union
import math

class BatchFrameProcessor:
    """
    A parallel frame processor that uses batch processing to avoid OVITO pickle issues.
    
    This class processes molecular dynamics frames in parallel by grouping them into batches,
    where each batch is processed by a single worker process. This approach avoids the
    serialization issues with OVITO Pipeline objects while maintaining efficiency.
    
    Example:
        >>> processor = BatchFrameProcessor(
        ...     filename="simulation.dump",
        ...     output_repo="results/"
        ... )
        >>> frames = list(range(1, 1001, 50))
        >>> results = processor.process_frames_parallel(frames, num_batches=4)
    """
    
    def __init__(self, filename: str, output_repo: str, delta_gamma: float = 5, 
                 max_dist: float = 100, wall_max_z: float = 4.8, 
                 delta_y_axis: float = 1, type: str = 'spherical', 
                 particle_type_wall: set = {1}, 
                 particule_liquid_type: set = {1, 2},
                 oxygen_type: float = 3,
                 hydrogen_type: float = 2):
        """
        Initialize the BatchFrameProcessor.
        
        Args:
            filename (str): Path to the dump file
            output_repo (str): Output directory for results
            delta_gamma (float): Gamma parameter for contact angle calculation
            max_dist (float): Maximum distance for analysis
            wall_max_z (float): Maximum Z coordinate for wall
            delta_y_axis (float): Y-axis delta parameter
            type (str): Type of analysis ('spherical' or other)
            particle_type_wall (set): Set of particle types for wall
            particule_liquid_type (set): Set of particle types for liquid
            value_dist_marge_wall_liquide (float): Distance margin for wall-liquid
            oxygen_type (float): Oxygen particle type identifier
            hydrogen_type (float): Hydrogen particle type identifier
        """
        self.filename = filename
        self.output_repo = output_repo
        self.delta_gamma = delta_gamma
        self.particule_liquid_type = particule_liquid_type
        self.max_dist = max_dist
        self.wall_max_z = wall_max_z
        self.delta_y_axis = delta_y_axis
        self.type = type
        self.particle_type_wall = particle_type_wall
        self.oxygen_type = oxygen_type
        self.hydrogen_type = hydrogen_type
        
        # Ensure output directory exists
        os.makedirs(self.output_repo, exist_ok=True)
    
    def process_frames_parallel(self, frames_to_process: List[int], 
                              num_batches: int = 4, 
                              max_workers: Optional[int] = None) -> Dict[int, float]:
        """
        Process multiple frames in parallel using batch processing.
        
        This method divides frames into batches and processes each batch in a separate
        worker process. This avoids OVITO serialization issues while maintaining
        parallel efficiency.
        
        Args:
            frames_to_process (List[int]): List of frame numbers to process
            num_batches (int): Number of batches to create (default: 4)
            max_workers (Optional[int]): Maximum worker processes (default: num_batches)
        
        Returns:
            Dict[int, float]: Dictionary mapping frame numbers to mean contact angles
            
        Example:
            >>> frames = [1, 51, 101, 151, 201]
            >>> results = processor.process_frames_parallel(frames, num_batches=2)
            >>> print(f"Processed {len(results)} frames")
        """
        if max_workers is None:
            max_workers = num_batches
        
        # Create batches
        batches = self._create_batches(frames_to_process, num_batches)
        print(f"Processing {len(frames_to_process)} frames in {len(batches)} batches "
              f"with {max_workers} workers")
        
        # Process batches in parallel
        results = {}
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches
            future_to_batch = {}
            for i, batch_frames in enumerate(batches):
                future = executor.submit(self._process_batch_worker, batch_frames)
                future_to_batch[future] = i
            
            # Collect results as they complete
            completed_batches = 0
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_results = future.result()
                    completed_batches += 1
                    print(f"Completed batch {completed_batches}/{len(batches)} "
                          f"({len(batch_results)} frames)")
                    
                    # Add results to main dictionary
                    for frame_num, mean_alpha in batch_results:
                        if mean_alpha is not None:
                            results[frame_num] = mean_alpha
                            
                except Exception as e:
                    print(f"Error in batch {batch_idx + 1}: {e}")
        
        print(f"Successfully processed {len(results)}/{len(frames_to_process)} frames")
        return results
    
    def process_frame_single(self, frame_num: int) -> Tuple[int, float]:
        """
        Process a single frame (for testing or sequential processing).
        
        Args:
            frame_num (int): Frame number to process
            
        Returns:
            Tuple[int, float]: Frame number and mean contact angle
            
        Example:
            >>> frame_num, mean_angle = processor.process_frame_single(1)
            >>> print(f"Frame {frame_num}: {mean_angle:.2f}°")
        """
        return self._process_single_frame(frame_num)
    
    def _create_batches(self, frames: List[int], num_batches: int) -> List[List[int]]:
        """
        Split frames into approximately equal batches.
        
        Args:
            frames (List[int]): List of frame numbers
            num_batches (int): Number of batches to create
            
        Returns:
            List[List[int]]: List of batches
        """
        if num_batches >= len(frames):
            # If we have more batches than frames, one frame per batch
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
        
        This function is called by each worker process and initializes the parsers
        once per batch to avoid repeated file parsing overhead.
        
        Args:
            batch_frames (List[int]): List of frame numbers in this batch
            
        Returns:
            List[Tuple[int, Optional[float]]]: Results for each frame in batch
        """
        # Initialize parsers once per batch
        water_parser = WaterOxygenDumpParser(
            self.filename, 
            self.particle_type_wall, 
            self.oxygen_type, 
            self.hydrogen_type
        )
        wall_parser = DumpParse_wall(self.filename, self.particule_liquid_type)
        
        batch_results = []
        
        for frame_num in batch_frames:
            try:
                result = self._process_single_frame_with_parsers(
                    frame_num, water_parser, wall_parser
                )
                batch_results.append(result)
                
            except Exception as e:
                print(f"Error processing frame {frame_num}: {e}")
                batch_results.append((frame_num, None))
        
        return batch_results
    
    def _process_single_frame(self, frame_num: int) -> Tuple[int, float]:
        """Process a single frame with new parser instances."""
        water_parser = WaterOxygenDumpParser(
            self.filename, 
            self.particle_type_wall, 
            self.oxygen_type, 
            self.hydrogen_type
        )
        wall_parser = DumpParse_wall(self.filename, self.particule_liquid_type)
        
        return self._process_single_frame_with_parsers(frame_num, water_parser, wall_parser)
    
    def _process_single_frame_with_parsers(self, frame_num: int, 
                                         water_parser: WaterOxygenDumpParser, 
                                         wall_parser: DumpParse_wall) -> Tuple[int, float]:
        """
        Process a single frame using provided parser instances.
        
        Args:
            frame_num (int): Frame number to process
            water_parser (WaterOxygenDumpParser): Parser for water oxygen data
            wall_parser (DumpParse_wall): Parser for wall data
            
        Returns:
            Tuple[int, float]: Frame number and mean contact angle
        """
        # Parse data for this frame using the new WaterOxygenDumpParser methods
        # First parse the frame to load the data
        water_parser.parse(num_frame=frame_num)
        # Get water oxygen positions using the specialized method
        parsed_xyz = water_parser.get_water_oxygen_positions(frame_num)

        # Calculate mean position
        mean_parsed = np.mean(parsed_xyz, axis=0)
        
        # Create predictor and calculate contact angles
        classpredictor = ContactAnglePredictor(
            parsed_xyz, 
            self.delta_gamma, 
            self.max_dist, 
            mean_parsed, 
            10, 
            self.delta_y_axis,
            type=self.type
        )
        
        list_1, list_2, list_3 = classpredictor.predict_contact_angle()
        mean_alpha = np.mean(list_1)
        
        # Save results
        np.savetxt(f"{self.output_repo}/alfasframe{frame_num}.txt", 
                  np.array(list_1), fmt='%f')
        np.save(f"{self.output_repo}/surfacesframe{frame_num}.npy", 
               np.array(list_2))
        np.save(f"{self.output_repo}/poptsframe{frame_num}.npy", 
               np.array(list_3))
        
        print(f"Frame {frame_num} - mean angle: {mean_alpha:.2f}°")
        
        return frame_num, mean_alpha
    
    def get_batch_info(self, frames_to_process: List[int], num_batches: int) -> Dict:
        """
        Get information about how frames will be distributed across batches.
        
        Args:
            frames_to_process (List[int]): List of frame numbers
            num_batches (int): Number of batches
            
        Returns:
            Dict: Information about batch distribution
            
        Example:
            >>> info = processor.get_batch_info([1, 51, 101, 151, 201], 2)
            >>> print(f"Batch sizes: {info['batch_sizes']}")
        """
        batches = self._create_batches(frames_to_process, num_batches)
        
        return {
            'total_frames': len(frames_to_process),
            'num_batches': len(batches),
            'batch_sizes': [len(batch) for batch in batches],
            'frames_per_batch': [batch for batch in batches],
            'max_batch_size': max(len(batch) for batch in batches),
            'min_batch_size': min(len(batch) for batch in batches)
        }


class GPUFrameProcessor(FrameProcessor):
    def process_frame(self, frame_num, ):
        """
        Process a single frame using GPU acceleration to calculate contact angles and save results.
        
        Args:
            frame_num (int): Frame number to process.
        
        Returns:
            tuple: Frame number and mean contact angle.
        """
        try:
            import cupy as cp
        except ImportError:
            print("CuPy not found. Falling back to CPU processing.")
            return super().process_frame(frame_num)
            
        file = DumpParser(self.filename,  self.particle_type_wall)
        parsed = file.parse(frame_num)
        
        # Transfer data to GPU
        try:
            parsed_gpu = cp.array(parsed)
            mean_parsed = cp.mean(parsed_gpu, axis=0)
            mean_parsed_cpu = cp.asnumpy(mean_parsed)
            parsed_cpu = cp.asnumpy(parsed_gpu)
            
            # Free GPU memory
            del parsed_gpu
            cp.get_default_memory_pool().free_all_blocks()
            
        except Exception as e:
            print(f"GPU processing failed: {e}. Falling back to CPU.")
            mean_parsed_cpu = np.mean(parsed, axis=0)
            parsed_cpu = parsed
        
        classpredictor = ContactAnglePredictor(
            parsed_cpu, self.delta_gamma, self.max_dist, mean_parsed_cpu, 
            self.wall_max_z, 10, self.delta_y_axis, type=self.type
        )
        
        list_1, list_2, list_3 = classpredictor.predict_contact_angle()
        mean_alpha = np.mean(list_1)
        
        np.savetxt(f"{self.output_repo}/alfasframe{frame_num}.txt", np.array(list_1), fmt='%f')
        np.save(f"{self.output_repo}/surfacesframe{frame_num}.npy", np.array(list_2))
        np.save(f"{self.output_repo}/poptsframe{frame_num}.npy", np.array(list_3))
        print(f"Frame {frame_num} - mean angle: {mean_alpha}")
        
        return frame_num, mean_alpha