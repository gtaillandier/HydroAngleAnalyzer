import math
import os
import numpy as np

class frames_angle_PostProcessor:
    def __init__(self):
        pass

    def undump(self, dump_id):
        """
        Simulate the undump command in LAMMPS to stop a dump.

        :param dump_id: The identifier of the dump to stop.
        :return: A message indicating the dump has been stopped.
        """
        return f"undump {dump_id}"

    def calculate_overall_mean_and_std(self, csv_file_path):
        """
        Calculate the overall mean and standard deviation from a CSV file.

        :param csv_file_path: Path to the CSV file.
        :return: A tuple containing the overall mean and standard deviation.
        """
        mean_values = []

        with open(csv_file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # Skip the header row

            for row in csv_reader:
                if len(row) >= 2:  # Ensure there are at least two columns
                    mean_value = float(row[1])
                    mean_values.append(mean_value)

        if not mean_values:
            return None, None

        # Calculate the overall mean
        overall_mean = sum(mean_values) / len(mean_values)

        # Calculate the standard deviation
        squared_diffs = [(x - overall_mean) ** 2 for x in mean_values]
        variance = sum(squared_diffs) / len(mean_values)
        std_dev = math.sqrt(variance)

        return overall_mean, std_dev

    def calculate_statistics_from_file(self, file_path):
        """
        Calculate the mean and median from a file.

        :param file_path: Path to the file.
        :return: A tuple containing the mean and median values.
        """
        with open(file_path, 'r') as file:
            values = [float(line.strip()) for line in file if line.strip()]

        if not values:
            return None, None

        mean_value = np.mean(values)
        median_value = np.median(values)

        return mean_value, median_value

    def generate_csv(self, directory_path, output_mean_csv_path, output_median_csv_path):
        """
        Generate CSV files containing mean and median values from files in a directory.

        :param directory_path: Path to the directory containing the files.
        :param output_mean_csv_path: Path for the mean values CSV file.
        :param output_median_csv_path: Path for the median values CSV file.
        """
        with open(output_mean_csv_path, 'w', newline='') as mean_csv_file, \
             open(output_median_csv_path, 'w', newline='') as median_csv_file:

            mean_csv_writer = csv.writer(mean_csv_file)
            median_csv_writer = csv.writer(median_csv_file)

            mean_csv_writer.writerow(['File Number', 'Mean Value'])
            median_csv_writer.writerow(['File Number', 'Median Value'])

            for i in range(1, 1000):
                file_name = f'alfasframe{i}.txt'
                file_path = os.path.join(directory_path, file_name)

                if os.path.exists(file_path):
                    mean_value, median_value = self.calculate_statistics_from_file(file_path)

                    if mean_value is not None and median_value is not None:
                        mean_csv_writer.writerow([i, mean_value])
                        median_csv_writer.writerow([i, median_value])

# # Example usage
# if __name__ == "__main__":
#     post_processor = LAMMPSPostProcessor()

#     # Example usage of the undump method
#     print(post_processor.undump("my_dump"))

#     # Example usage of the CSV processing methods
#     directory_path = 'out_parall_unfix_graphite2/'  # Replace with the path to your repo
#     output_mean_csv_path = 'output_mean_fix.csv'  # Path for the mean values CSV file
#     output_median_csv_path = 'output_median_fix.csv'  # Path for the median values CSV file

#     post_processor.generate_csv(directory_path, output_mean_csv_path, output_median_csv_path)

#     csv_file_path = 'output_mean_fix.csv'  # Replace with your CSV file path
#     overall_mean, std_dev = post_processor.calculate_overall_mean_and_std(csv_file_path)

#     if overall_mean is not None and std_dev is not None:
#         print(f"Overall Mean: {overall_mean}")
#         print(f"Standard Deviation: {std_dev}")
#     else:
#         print("No data available to calculate mean and standard deviation.")