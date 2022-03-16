#   Copyright (C) IMDEA Networks Institute 2022
#   This program is free software: you can redistribute it and/or modify
#
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see http://www.gnu.org/licenses/.
#
#   Authors: Yago Lizarribar <yago.lizarribar [at] imdea [dot] org>
#

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
