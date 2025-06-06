import numpy as np
import os
import psutil
import gc
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from hydroangleanalyzer import ContactAnglePredictor, DumpParser
import multiprocessing as mp
from functools import partial
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HighPerformanceFrameProcessor:
    def __init__(self, filename, output_repo, delta_gamma=5, max_dist=100, wall_max_z=4.8, 
                 delta_y_axis=1, type='spherical', particle_type_wall={2, 3}):
        self.filename = filename
        self.output_repo = output_repo
        self.delta_gamma = delta_gamma
        self.max_dist = max_dist
        self.wall_max_z = wall_max_z
        self.delta_y_axis = delta_y_axis
        self.type = type
        self.particle_type_wall = particle_type_wall
        
        os.makedirs(self.output_repo, exist_ok=True)
        
        # System info for optimization
        self.available_memory_gb = psutil.virtual_memory().available / (1024**3)
        self.cpu_count = mp.cpu_count()
        logger.info(f"Available memory: {self.available_memory_gb:.1f} GB")
        logger.info(f"CPU cores: {self.cpu_count}")
        
        # Initialize parser once
        self.file_parser = None
        self._initialize_parser()
    
    def _initialize_parser(self):
        """Initialize the parser once to avoid repeated file reading."""
        logger.info("Initializing DumpParser for large file...")
        start_time = time.time()
        self.file_parser = DumpParser(self.filename, self.particle_type_wall)
        init_time = time.time() - start_time
        logger.info(f"DumpParser initialized in {init_time:.2f} seconds")
    
    def estimate_memory_usage(self, sample_frames=5):
        """Estimate memory usage by sampling a few frames."""
        logger.info("Estimating memory requirements...")
        
        sample_frames = min(sample_frames, 5)  # Sample up to 5 frames
        total_size = 0
        
        for i in range(sample_frames):
            frame_num = i * 100  # Sample every 100th frame
            try:
                parsed = self.file_parser.parse(frame_num)
                frame_size = parsed.nbytes
                total_size += frame_size
                logger.info(f"Frame {frame_num}: {frame_size / (1024**2):.1f} MB")
            except:
                continue
        
        if total_size > 0:
            avg_frame_size_mb = (total_size / sample_frames) / (1024**2)
            total_estimated_gb = (avg_frame_size_mb * 5000) / 1024
            logger.info(f"Average frame size: {avg_frame_size_mb:.1f} MB")
            logger.info(f"Estimated total size for 5000 frames: {total_estimated_gb:.1f} GB")
            
            return avg_frame_size_mb, total_estimated_gb
        
        return 0, 0
    
    def process_frame_batch(self, frame_numbers, progress_callback=None):
        """Process a batch of frames efficiently."""
        results = []
        
        for i, frame_num in enumerate(frame_numbers):
            try:
                result = self._process_single_frame(frame_num)
                results.append(result)
                
                if progress_callback:
                    progress_callback(frame_num, i + 1, len(frame_numbers))
                    
            except Exception as e:
                logger.error(f"Error processing frame {frame_num}: {e}")
                results.append((frame_num, None))
        
        return results
    
    def _process_single_frame(self, frame_num):
        """Process a single frame efficiently."""
        # Parse frame data
        parsed = self.file_parser.parse(frame_num)
        mean_parsed = np.mean(parsed, axis=0)
        
        # Create predictor
        classpredictor = ContactAnglePredictor(
            parsed, self.delta_gamma, self.max_dist, mean_parsed,
            self.wall_max_z, 10, self.delta_y_axis, type=self.type
        )
        
        # Predict contact angles
        list_1, list_2, list_3 = classpredictor.predict_contact_angle()
        mean_alpha = np.mean(list_1)
        
        # Save results efficiently
        self._save_frame_results(frame_num, list_1, list_2, list_3)
        
        # Clean up large objects immediately
        del parsed, classpredictor, list_2, list_3
        
        return frame_num, mean_alpha
    
    def _save_frame_results(self, frame_num, list_1, list_2, list_3):
        """Save frame results with optimized I/O."""
        # Use numpy's faster saving methods
        alpha_path = f"{self.output_repo}/alfasframe{frame_num}.txt"
        surfaces_path = f"{self.output_repo}/surfacesframe{frame_num}.npy"
        popts_path = f"{self.output_repo}/poptsframe{frame_num}.npy"
        
        # Save data
        np.savetxt(alpha_path, np.array(list_1), fmt='%f')
        np.save(surfaces_path, np.array(list_2))
        np.save(popts_path, np.array(list_3))
    
    def process_all_frames_chunked(self, total_frames=5000, chunk_size=None, 
                                 start_frame=0, progress_interval=100):
        """
        Process all frames in chunks to manage memory efficiently.
        """
        if chunk_size is None:
            # Estimate optimal chunk size based on available memory
            avg_frame_mb, total_gb = self.estimate_memory_usage()
            if avg_frame_mb > 0:
                # Use 25% of available memory for chunk processing
                target_memory_gb = self.available_memory_gb * 0.25
                chunk_size = max(1, int((target_memory_gb * 1024) / avg_frame_mb))
                chunk_size = min(chunk_size, 200)  # Cap at 200 frames per chunk
            else:
                chunk_size = 50  # Conservative default
        
        logger.info(f"Processing {total_frames} frames in chunks of {chunk_size}")
        
        results = []
        processed_count = 0
        
        for chunk_start in range(start_frame, start_frame + total_frames, chunk_size):
            chunk_end = min(chunk_start + chunk_size, start_frame + total_frames)
            frame_numbers = list(range(chunk_start, chunk_end))
            
            logger.info(f"Processing chunk: frames {chunk_start}-{chunk_end-1}")
            
            def progress_callback(frame_num, current, total):
                nonlocal processed_count
                processed_count += 1
                if processed_count % progress_interval == 0:
                    percent = (processed_count / total_frames) * 100
                    logger.info(f"Progress: {processed_count}/{total_frames} ({percent:.1f}%)")
            
            chunk_results = self.process_frame_batch(frame_numbers, progress_callback)
            results.extend(chunk_results)
            
            # Force garbage collection between chunks
            gc.collect()
            
            # Log memory usage
            memory_percent = psutil.virtual_memory().percent
            logger.info(f"Memory usage after chunk: {memory_percent:.1f}%")
        
        logger.info(f"Completed processing {len(results)} frames")
        return results
    
    def process_frames_parallel_conservative(self, frame_numbers, max_workers=None):
        """
        Conservative parallel processing with memory management.
        Uses threading to avoid parser reinitialization overhead.
        """
        if max_workers is None:
            # Conservative threading for large data
            max_workers = min(3, self.cpu_count // 2)
        
        logger.info(f"Using {max_workers} threads for parallel processing")
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_frame = {
                executor.submit(self._process_single_frame, frame_num): frame_num 
                for frame_num in frame_numbers
            }
            
            # Process completed tasks
            for future in as_completed(future_to_frame):
                frame_num = future_to_frame[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if len(results) % 100 == 0:
                        percent = (len(results) / len(frame_numbers)) * 100
                        logger.info(f"Completed {len(results)}/{len(frame_numbers)} frames ({percent:.1f}%)")
                        
                except Exception as e:
                    logger.error(f"Frame {frame_num} failed: {e}")
                    results.append((frame_num, None))
        
        return sorted(results, key=lambda x: x[0])


# Multiprocessing approach for maximum performance (if parser can be recreated efficiently)
def process_frame_worker(args):
    """Worker function for multiprocessing approach."""
    frame_num, filename, output_repo, delta_gamma, max_dist, wall_max_z, delta_y_axis, type_param, particle_type_wall = args
    
    try:
        # Create parser for this process
        file_parser = DumpParser(filename, particle_type_wall)
        parsed = file_parser.parse(frame_num)
        mean_parsed = np.mean(parsed, axis=0)
        
        # Process frame
        classpredictor = ContactAnglePredictor(
            parsed, delta_gamma, max_dist, mean_parsed, wall_max_z, 10, delta_y_axis, type=type_param
        )
        
        list_1, list_2, list_3 = classpredictor.predict_contact_angle()
        mean_alpha = np.mean(list_1)
        
        # Save results
        np.savetxt(f"{output_repo}/alfasframe{frame_num}.txt", np.array(list_1), fmt='%f')
        np.save(f"{output_repo}/surfacesframe{frame_num}.npy", np.array(list_2))
        np.save(f"{output_repo}/poptsframe{frame_num}.npy", np.array(list_3))
        
        return frame_num, mean_alpha
        
    except Exception as e:
        logger.error(f"Error in worker for frame {frame_num}: {e}")
        return frame_num, None


class MultiProcessingFrameProcessor:
    def __init__(self, filename, output_repo, delta_gamma=5, max_dist=100, wall_max_z=4.8,
                 delta_y_axis=1, type='spherical', particle_type_wall={2, 3}):
        self.filename = filename
        self.output_repo = output_repo
        self.delta_gamma = delta_gamma
        self.max_dist = max_dist
        self.wall_max_z = wall_max_z
        self.delta_y_axis = delta_y_axis
        self.type = type
        self.particle_type_wall = particle_type_wall
        os.makedirs(self.output_repo, exist_ok=True)
    
    def process_frames_multiprocessing(self, frame_numbers, max_workers=None, chunk_size=10):
        """
        Process frames using multiprocessing with chunked submission.
        """
        if max_workers is None:
            max_workers = min(mp.cpu_count() - 1, 8)  # Leave one core free, cap at 8
        
        logger.info(f"Using {max_workers} processes for multiprocessing")
        
        # Prepare arguments
        args_list = [
            (frame_num, self.filename, self.output_repo, self.delta_gamma,
             self.max_dist, self.wall_max_z, self.delta_y_axis, self.type, self.particle_type_wall)
            for frame_num in frame_numbers
        ]
        
        results = []
        
        # Process in chunks to manage memory
        for i in range(0, len(args_list), chunk_size * max_workers):
            chunk = args_list[i:i + chunk_size * max_workers]
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                chunk_results = list(executor.map(process_frame_worker, chunk))
                results.extend(chunk_results)
                
                completed = i + len(chunk)
                percent = (completed / len(frame_numbers)) * 100
                logger.info(f"Completed {completed}/{len(frame_numbers)} frames ({percent:.1f}%)")
        
        return sorted(results, key=lambda x: x[0])
