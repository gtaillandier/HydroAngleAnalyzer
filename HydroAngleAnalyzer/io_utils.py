from ovito.io import import_file

def load_dump_ovito(in_path):
    pipeline = import_file(in_path)
    # Add necessary modifiers
    return pipeline

def save_array_as_txt(array, filename):
    np.savetxt(filename, array, fmt='%f')
