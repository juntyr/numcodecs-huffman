"""
[`HuffmanCodec`][numcodecs_huffman.HuffmanCodec] for the [`numcodecs`][numcodecs] buffer compression API.
"""

__all__ = ["HuffmanCodec"]

from io import BytesIO
from typing import Any, TypeVar

import numcodecs.compat
import numcodecs.registry
import numpy as np
import varint
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
        message = []

        message.append(varint.encode(len(dtype.str)))
        message.append(dtype.str.encode("ascii"))

        message.append(varint.encode(len(shape)))
        for s in shape:
            message.append(varint.encode(s))

        table = huffman.get_code_table()
        table_no_eof = [
            (k, e) for k, e in huffman.get_code_table().items() if k != _EOF
        ]
        message.append(varint.encode(len(table_no_eof)))
        # FIXME: what about endianness
        message.append(np.array([k for k, _ in table_no_eof]).tobytes())
        for k, (bitsize, value) in table_no_eof:
            message.append(varint.encode(bitsize))
            message.append(varint.encode(value))
        bitsize, value = table[_EOF]
        message.append(varint.encode(bitsize))
        message.append(varint.encode(value))

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

        dtype = np.dtype(b_io.read(varint.decode_stream(b_io)).decode("ascii"))

        shape = tuple(
            varint.decode_stream(b_io) for _ in range(varint.decode_stream(b_io))
        )

        table_len = varint.decode_stream(b_io)
        table_keys = np.frombuffer(
            b_io.read(table_len * dtype.itemsize), dtype=dtype, count=table_len
        )
        table = dict()
        for k in table_keys:
            table[k] = (varint.decode_stream(b_io), varint.decode_stream(b_io))
        table[_EOF] = (varint.decode_stream(b_io), varint.decode_stream(b_io))
        huffman = DaHuffmanCodec(table)

        decoded = np.array(huffman.decode(b_io.read())).reshape(shape)

        return numcodecs.compat.ndarray_copy(decoded, out)  # type: ignore


numcodecs.registry.register_codec(HuffmanCodec)


def _as_bits(a: np.ndarray[S, np.dtype[Any]], /) -> np.ndarray[S, np.dtype[Any]]:
    """
    Reinterprets the array `a` to its binary representation.

    Parameters
    ----------
    a : np.ndarray[S, np.dtype[Any]]
        The array to reinterpret as binary.

    Returns
    -------
    binary : np.ndarray[S, np.dtype[Any]]
        The binary representation of the array `a`.
    """

    return a.view(a.dtype.str.replace("f", "u").replace("i", "u"))  # type: ignore
