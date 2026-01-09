"""
[`HuffmanCodec`][numcodecs_huffman.HuffmanCodec] for the [`numcodecs`][numcodecs] buffer compression API.
"""

__all__ = ["HuffmanCodec"]

from io import BytesIO
from typing import Any, TypeVar

import leb128
import numcodecs.compat
import numcodecs.registry
import numpy as np
from dahuffman import HuffmanCodec as DaHuffmanCodec
from dahuffman.huffmancodec import _EOF
from numcodecs.abc import Codec
from typing_extensions import Buffer  # MSPV 3.12

S = TypeVar("S", bound=tuple[int, ...])
""" Any array shape. """


class HuffmanCodec(Codec):
    """
    Codec that uses Huffman entropy coding to encode the data.

    Encoding produces a bytestring containing the Huffman code table and the
    encoded data.
    """

    __slots__ = ()

    codec_id: str = "huffman"  # type: ignore

    def encode(self, buf: Buffer) -> bytes:
        """
        Encode the data in `buf`.

        Parameters
        ----------
        buf : Buffer
            Data to be encoded. May be any object supporting the new-style
            buffer protocol.

        Returns
        -------
        enc : bytes
            Encoded data as a bytestring.
        """

        a = numcodecs.compat.ensure_ndarray(buf)
        dtype, shape = a.dtype, a.shape
        a = _as_bits(a.flatten())

        huffman = DaHuffmanCodec.from_data(a)
        encoded = huffman.encode(a)

        # message: dtype shape table encoded
        message: list[bytes | bytearray] = []

        message.append(leb128.u.encode(len(dtype.str)))
        message.append(dtype.str.encode("ascii"))

        message.append(leb128.u.encode(len(shape)))
        for s in shape:
            message.append(leb128.u.encode(s))

        table = huffman.get_code_table()
        table_no_eof = [
            (k, e) for k, e in huffman.get_code_table().items() if k != _EOF
        ]
        message.append(leb128.u.encode(len(table_no_eof)))

        # ensure that the table keys are encoded in little endian binary
        table_keys_array = np.array([k for k, _ in table_no_eof])
        message.append(
            table_keys_array.astype(table_keys_array.dtype.newbyteorder("<")).tobytes()
        )

        for k, (bitsize, value) in table_no_eof:
            message.append(leb128.u.encode(bitsize))
            message.append(leb128.u.encode(value))
        bitsize, value = table[_EOF]
        message.append(leb128.u.encode(bitsize))
        message.append(leb128.u.encode(value))

        message.append(encoded)

        return b"".join(message)

    def decode(self, buf: Buffer, out: None | Buffer = None) -> Buffer:
        """
        Decode the data in `buf`.

        Parameters
        ----------
        buf : Buffer
            Encoded data. Must be an object representing a bytestring, e.g.
            [`bytes`][bytes] or a 1D array of [`np.uint8`][numpy.uint8]s etc.
        out : Buffer, optional
            Writeable buffer to store decoded data. N.B. if provided, this
            buffer must be exactly the right size to store the decoded data.

        Returns
        -------
        dec : Buffer
            Decoded data. May be any object supporting the new-style buffer
            protocol.
        """

        b = numcodecs.compat.ensure_bytes(buf)

        b_io = BytesIO(b)

        dtype = np.dtype(b_io.read(leb128.u.decode_reader(b_io)[0]).decode("ascii"))

        shape = tuple(
            leb128.u.decode_reader(b_io)[0]
            for _ in range(leb128.u.decode_reader(b_io)[0])
        )

        table_len, _ = leb128.u.decode_reader(b_io)

        # decode the table keys from little endian binary
        # change them back to dtype_bits byte order
        table_keys = np.frombuffer(
            b_io.read(table_len * dtype.itemsize),
            dtype=_dtype_bits(dtype).newbyteorder("<"),
            count=table_len,
        )

        table = dict()
        for k in table_keys:
            table[k] = (
                leb128.u.decode_reader(b_io)[0],
                leb128.u.decode_reader(b_io)[0],
            )
        if len(table) > 0:
            table[_EOF] = (
                leb128.u.decode_reader(b_io)[0],
                leb128.u.decode_reader(b_io)[0],
            )
        huffman = DaHuffmanCodec(table)

        decoded = (
            np.array(huffman.decode(b_io.read()))
            .astype(_dtype_bits(dtype))
            .view(dtype)
            .reshape(shape)
        )

        return numcodecs.compat.ndarray_copy(decoded, out)  # type: ignore


numcodecs.registry.register_codec(HuffmanCodec)


def _as_bits(a: np.ndarray[S, np.dtype[Any]], /) -> np.ndarray[S, np.dtype[Any]]:
    """
    Reinterprets the array `a` to its binary (unsigned integer) representation.

    Parameters
    ----------
    a : np.ndarray[S, np.dtype[Any]]
        The array to reinterpret as binary.

    Returns
    -------
    binary : np.ndarray[S, np.dtype[Any]]
        The binary representation of the array `a`.
    """

    return a.view(_dtype_bits(a.dtype))  # type: ignore


def _dtype_bits(dtype: np.dtype) -> np.dtype:
    """
    Converts the `dtype` to its binary (unsigned integer) representation.

    Parameters
    ----------
    dtype : np.dtype
        The dtype to convert.

    Returns
    -------
    binary : np.dtype
        The binary dtype with equivalent size and alignment but unsigned
        integer kind.
    """

    return np.dtype(dtype.str.replace("f", "u").replace("i", "u"))
