from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Data import IUPACData
import io

#By Tyler Mullins for Evolution Class

# Paste your FASTA content here
fasta_input = """\
>NC_000024.10:c2787682-2786855 Homo sapiens chromosome Y, GRCh38.p14 Primary Assembly
AGAAGTGAGTTTTGGATAGTAAAATAAGTTTCGAACTCTGGCACCTTTCAATTTTGTCGCACTCTCCTTG
TTTTTGACAATGCAATCATATGCTTCTGCTATGTTAAGCGTATTCAACAGCGATGATTACAGTCCAGCTG
TGCAAGAGAATATTCCCGCTCTCCGGAGAAGCTCTTCCTTCCTTTGCACTGAAAGCTGTAACTCTAAGTA
TCAGTGTGAAACGGGAGAAAACAGTAAAGGCAACGTCCAGGATAGAGTGAAGCGACCCATGAACGCATTC
ATCGTGTGGTCTCGCGATCAGAGGCGCAAGATGGCTCTAGAGAATCCCAGAATGCGAAACTCAGAGATCA
GCAAGCAGCTGGGATACCAGTGGAAAATGCTTACTGAAGCCGAAAAATGGCCATTCTTCCAGGAGGCACA
GAAATTACAGGCCATGCACAGAGAGAAATACCCGAATTATAAGTATCGACCTCGTCGGAAGGCGAAGATG
CTGCCGAAGAATTGCAGTTTGCTTCCCGCAGATCCCGCTTCGGTACTCTGCAGCGAAGTGCAACTGGACA
ACAGGTTGTACAGGGATGACTGTACGAAAGCCACACACTCAAGAATGGAGCACCAGCTAGGCCACTTACC
GCCCATCAACGCAGCCAGCTCACCGCAGCAACGGGACCGCTACAGCCACTGGACAAAGCTGTAGGACAAT
CGGGTAACATTGGCTACAAAGACCTACCTAGATGCTCCTTTTTACGATAACTTACAGCCCTCACTTTCTT
ATGTTTAGTTTCAATATTGTTTTCTTTTCTCTGGCTAATAAAGGCCTTATTCATTTCA
"""

# Convert the string to a file-like object
fasta_io = io.StringIO(fasta_input)
protein_records = []

# Translate sequences
for record in SeqIO.parse(fasta_io, "fasta"):
    dna_seq = record.seq
    # Trim sequence length to be divisible by 3
    if len(dna_seq) % 3 != 0:
        dna_seq = dna_seq[:len(dna_seq) - (len(dna_seq) % 3)]
    protein_seq = dna_seq.translate(to_stop=True)

    # Convert one-letter amino acids to three-letter codes
    three_letter_seq = '-'.join(
        IUPACData.protein_letters_1to3.get(aa, 'Xxx') for aa in str(protein_seq)
    )

    protein_records.append(
        SeqRecord(Seq(three_letter_seq), id=record.id, description="translated from DNA")
    )

# Print protein sequences in FASTA format
print("\nTranslated Protein Sequences:\n")
for record in protein_records:
    print(f">{record.id} {record.description}")
    print(str(record.seq))
