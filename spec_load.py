import numpy as np


def spec_load(filename, data_type="u8"):
    """
    Load IQ data from file

    Parameters:
    filename: Name of the file to process
    data_type: Format of the data inside the file. Supported types are u8 and f32

    Returns:
    numpy array of complex samples
    """

    if data_type == "u8":
        data = np.fromfile(filename, np.uint8)
        real = data[0::2]
        imag = data[1::2]

        min_length = min([len(real), len(imag)])
        real = real[:min_length]
        imag = imag[:min_length]

        real = (real - 127.0) / 128.0
        imag = (imag - 127.0) / 128.0

    elif data_type == "f32":
        data = np.fromfile(filename, np.float32)
        real = data[0::2]
        imag = data[1::2]

        min_length = min([len(real), len(imag)])
        real = real[:min_length]
        imag = imag[:min_length]

    else:
        raise ValueError(
            "Unsupported data type. Valid ones are Unsigned Integer (u8) and Float 32 (f32)"
        )

    return real + 1j * imag
