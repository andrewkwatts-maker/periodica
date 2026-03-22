"""
Enums for nucleic acid visualization and properties.
Includes DNA/RNA types, base types, modifications, and structure types.
"""

from enum import Enum


class NucleicAcidType(Enum):
    """Types of nucleic acids"""
    DNA = "dna"
    RNA = "rna"
    MRNA = "mrna"
    TRNA = "trna"
    RRNA = "rrna"
    MIRNA = "mirna"
    SIRNA = "sirna"
    SNRNA = "snrna"
    LNCRNA = "lncrna"

    @classmethod
    def from_string(cls, value):
        for member in cls:
            if member.value == value.lower():
                return member
        return cls.DNA

    @classmethod
    def get_color(cls, na_type):
        if isinstance(na_type, str):
            na_type = cls.from_string(na_type)
        colors = {
            cls.DNA: "#2196F3",      # Blue
            cls.RNA: "#4CAF50",      # Green
            cls.MRNA: "#66BB6A",     # Light green
            cls.TRNA: "#26A69A",     # Teal
            cls.RRNA: "#009688",     # Dark teal
            cls.MIRNA: "#FF9800",    # Orange
            cls.SIRNA: "#FF5722",    # Deep orange
            cls.SNRNA: "#9C27B0",    # Purple
            cls.LNCRNA: "#673AB7",   # Deep purple
        }
        return colors.get(na_type, "#9E9E9E")


class BaseType(Enum):
    """Nucleic acid base types"""
    ADENINE = "A"
    GUANINE = "G"
    CYTOSINE = "C"
    THYMINE = "T"      # DNA only
    URACIL = "U"       # RNA only
    INOSINE = "I"      # Modified base

    @classmethod
    def from_string(cls, value):
        for member in cls:
            if member.value == value.upper():
                return member
        return None

    @classmethod
    def get_color(cls, base):
        if isinstance(base, str):
            base = cls.from_string(base)
        colors = {
            cls.ADENINE: "#4CAF50",   # Green
            cls.GUANINE: "#FFC107",   # Amber
            cls.CYTOSINE: "#2196F3",  # Blue
            cls.THYMINE: "#F44336",   # Red
            cls.URACIL: "#FF5722",    # Deep orange
            cls.INOSINE: "#9C27B0",   # Purple
        }
        return colors.get(base, "#9E9E9E")

    @classmethod
    def get_complement(cls, base, is_rna=False):
        """Get Watson-Crick complement"""
        if isinstance(base, str):
            base = cls.from_string(base)
        complements_dna = {
            cls.ADENINE: cls.THYMINE,
            cls.THYMINE: cls.ADENINE,
            cls.GUANINE: cls.CYTOSINE,
            cls.CYTOSINE: cls.GUANINE,
        }
        complements_rna = {
            cls.ADENINE: cls.URACIL,
            cls.URACIL: cls.ADENINE,
            cls.GUANINE: cls.CYTOSINE,
            cls.CYTOSINE: cls.GUANINE,
        }
        if is_rna:
            return complements_rna.get(base)
        return complements_dna.get(base)


class SecondaryStructure(Enum):
    """Nucleic acid secondary structure types"""
    DOUBLE_HELIX = "double_helix"       # B-DNA, A-DNA
    SINGLE_STRAND = "single_strand"
    HAIRPIN = "hairpin"                  # Stem-loop
    BULGE = "bulge"
    INTERNAL_LOOP = "internal_loop"
    JUNCTION = "junction"               # 3-way or 4-way
    PSEUDOKNOT = "pseudoknot"
    G_QUADRUPLEX = "g_quadruplex"
    I_MOTIF = "i_motif"
    TRIPLE_HELIX = "triple_helix"

    @classmethod
    def from_string(cls, value):
        for member in cls:
            if member.value == value:
                return member
        return cls.SINGLE_STRAND

    @classmethod
    def get_color(cls, structure):
        if isinstance(structure, str):
            structure = cls.from_string(structure)
        colors = {
            cls.DOUBLE_HELIX: "#2196F3",
            cls.SINGLE_STRAND: "#9E9E9E",
            cls.HAIRPIN: "#4CAF50",
            cls.BULGE: "#FF9800",
            cls.INTERNAL_LOOP: "#FFC107",
            cls.JUNCTION: "#9C27B0",
            cls.PSEUDOKNOT: "#E91E63",
            cls.G_QUADRUPLEX: "#00BCD4",
            cls.I_MOTIF: "#3F51B5",
            cls.TRIPLE_HELIX: "#795548",
        }
        return colors.get(structure, "#9E9E9E")


class NucleicAcidFunction(Enum):
    """Nucleic acid functional categories"""
    GENETIC_STORAGE = "genetic_storage"
    PROTEIN_CODING = "protein_coding"
    AMINO_ACID_TRANSPORT = "amino_acid_transport"
    RIBOSOME_COMPONENT = "ribosome_component"
    GENE_REGULATION = "gene_regulation"
    SPLICING = "splicing"
    CATALYTIC = "catalytic"
    STRUCTURAL = "structural"
    SIGNALING = "signaling"

    @classmethod
    def from_string(cls, value):
        for member in cls:
            if member.value == value:
                return member
        return cls.GENETIC_STORAGE

    @classmethod
    def get_color(cls, func):
        if isinstance(func, str):
            func = cls.from_string(func)
        colors = {
            cls.GENETIC_STORAGE: "#2196F3",
            cls.PROTEIN_CODING: "#4CAF50",
            cls.AMINO_ACID_TRANSPORT: "#26A69A",
            cls.RIBOSOME_COMPONENT: "#009688",
            cls.GENE_REGULATION: "#FF9800",
            cls.SPLICING: "#9C27B0",
            cls.CATALYTIC: "#E91E63",
            cls.STRUCTURAL: "#795548",
            cls.SIGNALING: "#00BCD4",
        }
        return colors.get(func, "#9E9E9E")


class Modification(Enum):
    """Common nucleic acid modifications"""
    METHYLATION = "methylation"
    PSEUDOURIDINE = "pseudouridine"
    DIHYDROURIDINE = "dihydrouridine"
    INOSINE = "inosine"
    THIOURIDINE = "thiouridine"
    QUEUOSINE = "queuosine"
    WYBUTOSINE = "wybutosine"
    N6_METHYLADENOSINE = "m6A"
    PHOSPHOROTHIOATE = "phosphorothioate"

    @classmethod
    def from_string(cls, value):
        for member in cls:
            if member.value == value:
                return member
        return None


class NucleicAcidLayoutMode(Enum):
    """Layout modes for nucleic acid visualization"""
    GRID = "grid"
    TYPE = "type"
    FUNCTION = "function"
    LENGTH = "length"
    GC_CONTENT = "gc_content"
    ORGANISM = "organism"

    @classmethod
    def from_string(cls, value):
        for member in cls:
            if member.value == value:
                return member
        return cls.GRID


# Codon table for translation
CODON_TABLE = {
    # First position U
    'UUU': 'F', 'UUC': 'F', 'UUA': 'L', 'UUG': 'L',
    'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S',
    'UAU': 'Y', 'UAC': 'Y', 'UAA': '*', 'UAG': '*',
    'UGU': 'C', 'UGC': 'C', 'UGA': '*', 'UGG': 'W',
    # First position C
    'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',
    'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAU': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    # First position A
    'AUU': 'I', 'AUC': 'I', 'AUA': 'I', 'AUG': 'M',
    'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAU': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGU': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    # First position G
    'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',
    'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAU': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

# DNA codon table (T instead of U)
DNA_CODON_TABLE = {k.replace('U', 'T'): v for k, v in CODON_TABLE.items()}


def translate_sequence(mrna_sequence: str) -> str:
    """Translate mRNA sequence to protein sequence."""
    sequence = mrna_sequence.upper().replace('T', 'U')
    protein = []

    for i in range(0, len(sequence) - 2, 3):
        codon = sequence[i:i+3]
        aa = CODON_TABLE.get(codon, 'X')
        if aa == '*':  # Stop codon
            break
        protein.append(aa)

    return ''.join(protein)


def transcribe_dna(dna_sequence: str) -> str:
    """Transcribe DNA to mRNA (coding strand convention)."""
    return dna_sequence.upper().replace('T', 'U')


def reverse_complement(sequence: str, is_rna: bool = False) -> str:
    """Get reverse complement of a sequence."""
    if is_rna:
        complement_map = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
    else:
        complement_map = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}

    complement = ''.join(complement_map.get(base, 'N') for base in sequence.upper())
    return complement[::-1]


# Nucleotide molecular masses (Da)
NUCLEOTIDE_MASSES = {
    'A': 331.2,   # dAMP
    'T': 322.2,   # dTMP
    'G': 347.2,   # dGMP
    'C': 307.2,   # dCMP
    'U': 308.2,   # UMP (RNA)
}


NUCLEIC_ACID_PROPERTY_METADATA = {
    "length": {
        "display_name": "Length",
        "unit": "bp/nt",
        "min_value": 1,
        "max_value": 1000000,
    },
    "gc_content": {
        "display_name": "GC Content",
        "unit": "%",
        "min_value": 0,
        "max_value": 100,
    },
    "melting_temperature": {
        "display_name": "Melting Temperature",
        "unit": "°C",
        "min_value": 0,
        "max_value": 100,
    },
    "molecular_mass": {
        "display_name": "Molecular Mass",
        "unit": "kDa",
        "min_value": 0.1,
        "max_value": 10000,
    },
}
