"""One-hot encoding matching the reference ``spliceai2.dataset._one_hot_encode``."""
import numpy as np

NUM_CHANNELS = 14
NUCLEOTIDE_CHANNELS = {"A": 0, "C": 1, "G": 2, "T": 3}
ASSEMBLY_CHANNELS = {
    "GRCh38": 4,
    "NHGRI_mPanTro3-v2.0": 5,
    "Panubis1.0": 6,
    "T2T-MFA8v1.1": 7,
    "Mmul_10": 8,
    "ARS-UCD2.0": 9,
    "ARS-UI_Ramb_v3.0": 10,
    "Sscrofa11.1": 11,
    "GRCm38": 12,
    "GRCr8": 13,
}
HUMAN = "GRCh38"

# Base codes used for haplotype sequences: 0-3 = A, C, G, T; N_CODE for any
# other character; PAD_CODE for positions past a chromosome end.
N_CODE, PAD_CODE = 4, 5

# Like the reference, any other character (N, lowercase, ...) maps to channel 0.
_LUT = np.zeros(256, dtype=np.intp)
for _nuc, _channel in NUCLEOTIDE_CHANNELS.items():
    _LUT[ord(_nuc)] = _channel


def one_hot_encode(seq, assembly=HUMAN, length=None):
    """Returns a (14, length) uint8 array (length defaults to len(seq)).

    Positions past the end of ``seq`` are padding: all nucleotide channels
    zero, assembly channel set.
    """
    length = len(seq) if length is None else length
    codes = _LUT[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]
    x = np.zeros((NUM_CHANNELS, length), dtype=np.uint8)
    x[codes, np.arange(len(seq))] = 1
    x[ASSEMBLY_CHANNELS.get(assembly, 4)] = 1
    return x


def one_hot_codes(codes):
    """(14, n) uint8 one-hot of base codes, human assembly channel.

    N_CODE is encoded like ``one_hot_encode`` encodes N (channel 0);
    PAD_CODE like its padding (no nucleotide channel).
    """
    codes = np.asarray(codes)
    x = np.zeros((NUM_CHANNELS, len(codes)), dtype=np.uint8)
    nuc = np.nonzero(codes < 4)[0]
    x[codes[nuc], nuc] = 1
    x[0, codes == N_CODE] = 1
    x[ASSEMBLY_CHANNELS[HUMAN]] = 1
    return x


def random_sequences(n, length, seed=0):
    """n random ACGT strings; runtime does not depend on sequence content."""
    return random_sequences_of_lengths([length] * n, seed)


def random_sequences_of_lengths(lengths, seed=0):
    """One random ACGT string per entry in ``lengths``."""
    rng = np.random.default_rng(seed)
    alphabet = np.frombuffer(b"ACGT", dtype=np.uint8)
    return [
        alphabet[rng.integers(0, 4, length)].tobytes().decode("ascii")
        for length in lengths
    ]
