import numpy as np
from multiprocessing import get_context
import os
from hydro_angle_analyzer import ContactAnglePredictor, DumpParser

class FrameProcessor:
    def __init__(self, filename, output_repo, delta_gamma=5, max_dist=100, wall_max_z=4.8, delta_y_axis=1, type='spherical'):
        self.filename = filename
        self.output_repo = output_repo
        self.delta_gamma = delta_gamma
        self.max_dist = max_dist
        self.wall_max_z = wall_max_z
        self.delta_y_axis = delta_y_axis
        self.type = type

    def process_frame(self, frame_num):
        # Parse the frame using DumpParser
        file = DumpParser(self.filename)
        parsed = file.parse(frame_num)
        mean_parsed = np.mean(parsed, axis=0)

        # Initialize the ContactAnglePredictor
        classpredictor = ContactAnglePredictor(parsed, self.delta_gamma, self.max_dist, mean_parsed, self.wall_max_z, 10, self.delta_y_axis, type=self.type)

        # Calculate contact angle
        list_1, list_2, list_3 = classpredictor.predict_contact_angle()

        mean_alpha = np.mean(list_1)

        # Save results for each frame
        np.savetxt(f"{self.output_repo}/alfasframe{frame_num}.txt", np.array(list_1), fmt='%f')
        np.save(f"{self.output_repo}/surfacesframe{frame_num}.npy", np.array(list_2))
        np.save(f"{self.output_repo}/poptsframe{frame_num}.npy", np.array(list_3))
        print(f"Frame {frame_num} - mean angle: {mean_alpha}")

        return frame_num, mean_alpha

    def parallel_process_frames(self, frames):
        # Use multiprocessing with the 'spawn' start method
        num_processors = os.cpu_count()
        print('num process ava: ', num_processors)
        results = []
        with get_context("spawn").Pool(num_processors) as pool:
            futures = [
                pool.apply_async(self.process_frame, (frame,))
                for frame in frames
            ]
            for future in futures:
                try:
                    frame_num, mean_alpha = future.get()
                    results.append([frame_num, mean_alpha])
                except Exception as e:
                    print(f"Error processing frame: {e}")
        # Save all mean alphas to a single file after processing
        results = sorted(results, key=lambda x: x[0])
        np.savetxt(f"{self.output_repo}/alfas_per_frame_combined.txt", np.array(results), fmt='%f')
        print("Saved all mean alphas to 'alfas_per_frame_combined.txt'")
class GPUFrameProcessor(FrameProcessor):
    def process_frame(self, frame_num):
        # Parse the frame using DumpParser
        file = DumpParser(self.filename)
        parsed = file.parse(frame_num)
        mean_parsed = cp.mean(cp.array(parsed), axis=0)

        # Initialize the ContactAnglePredictor
        classpredictor = ContactAnglePredictor(cp.asnumpy(parsed), self.delta_gamma, self.max_dist, cp.asnumpy(mean_parsed), self.wall_max_z, 10, self.delta_y_axis, type=self.type)

        # Calculate contact angle
        list_1, list_2, list_3 = classpredictor.predict_contact_angle()

        mean_alpha = np.mean(list_1)

        # Save results for each frame
        np.savetxt(f"{self.output_repo}/alfasframe{frame_num}.txt", np.array(list_1), fmt='%f')
        np.save(f"{self.output_repo}/surfacesframe{frame_num}.npy", np.array(list_2))
        np.save(f"{self.output_repo}/poptsframe{frame_num}.npy", np.array(list_3))
        print(f"Frame {frame_num} - mean angle: {mean_alpha}")

        return frame_num, mean_alpha