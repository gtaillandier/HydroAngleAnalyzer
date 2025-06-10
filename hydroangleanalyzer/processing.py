import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
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
        self.file_parser = DumpParser(self.filename, self.particle_type_wall)
        self.parsed_wall = DumpParse_wall(self.filename, self.particule_liquid_type)
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
        parsed_xyz = self.file_parser.parse(frame_num)
        highest__part_wall = self.parsed_wall.find_highest_wall_part(frame_num)
        mean_parsed = np.mean(parsed_xyz, axis=0)
        
        classpredictor = ContactAnglePredictor(
            parsed_xyz, self.delta_gamma, self.max_dist, mean_parsed, self.wall_max_z, 10, self.delta_y_axis,  limit_dist_wall= highest__part_wall + self.value_dist_marge_wall_liquide , type=self.type
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