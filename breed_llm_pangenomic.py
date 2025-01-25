#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Extended Enhanced Pseudo-Pangenomic-LM Demo
------------------------------------------
This script:
1. Loads multiple reference genomes from FASTA files.
2. For each reference, chunk the sequence and store (chr_id, start, end) metadata.
3. Generates k-mers from each chunk, encodes them into embeddings, aggregates them,
   and appends assembly metadata features.
4. Stores the resulting embeddings and chunk-level metadata in a master list.
5. Handles a user query by generating k-mers for the query, computing an embedding,
   and comparing it to all chunk embeddings via similarity.
6. Retrieves the top matches, including the chromosome and position info, plus
   a similarity/confidence measure in the final annotation.
"""

import json
import torch
import numpy as np
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MinMaxScaler

# ----------------------------
# 1. Parameters and File Paths
# ----------------------------

# Define k-mer size
KMER_SIZE = 5      # Example k-mer size; adjust as needed

# Define how to chunk each reference sequence
CHUNK_SIZE = 5000  # e.g., 5 kb
OVERLAP = 1000     # Overlap between chunks

# List of reference genomes and their corresponding metadata files
# Example list of references using GCA accessions
#   - Add a 'sequence_report_path' if you have a separate file with JSON lines describing each chromosome
#   - Add a 'gbff_path' if you want to parse GenBank features from a .gbff file
genomes_info = [
    {
        "genome_path": "GCA_946409395.1_BROU-A-10.PacbioHiFiAssembly_genomic.fna",
        "metadata_path": "assembly_data_report_GCA_946409395.1_BROU-A-10.PacbioHiFiAssembly_genomic.jsonl",
        "sequence_report_path": "sequence_report_GCA_946409395.1_BROU-A-10.PacbioHiFiAssembly_genomic.jsonl",    # e.g. your file with chromosome-level JSON
        "gbff_path": "genomic_GCA_946409395.1_BROU-A-10.PacbioHiFiAssembly_genomic.gbff",
        "genome_id": "GCA_946409395.1"
    },
    {
        "genome_path": "GCA_946413285.1_IP-Hum-2.9549.PacbioHiFiAssembly_genomic.fna",
        "metadata_path": "assembly_data_report_GCA_946413285.1_IP-Hum-2.9549.PacbioHiFiAssembly_genomic.jsonl",
        "sequence_report_path": "sequence_report_GCA_946413285.1_IP-Hum-2.9549.PacbioHiFiAssembly_genomic.jsonl",  # If not available, just set None
        "gbff_path": "genomic_GCA_946413285.1_IP-Hum-2.9549.PacbioHiFiAssembly_genomic.gbff",
        "genome_id": "GCA_946413285.1"
    },
    
    {
        "genome_path": "GCA_964057265.1_LI-EF-011.685.PacbioHiFiAssembly_genomic.fna",
        "metadata_path": "assembly_data_report_GCA_964057265.1_LI-EF-011.685.PacbioHiFiAssembly_genomic.jsonl",
        "sequence_report_path": "sequence_report_GCA_964057265.1_LI-EF-011.685.PacbioHiFiAssembly_genomic.jsonl",  # If not available, just set None
        "gbff_path": "genomic_GCA_964057265.1_LI-EF-011.685.PacbioHiFiAssembly_genomic.gbff",
        "genome_id": "GCA_964057265.1"
    },
    # Add more genomes as needed ...
]

# ----------------------------
# 2. Helpers: sequence report and GenBank parsing
# ----------------------------

def load_sequence_report(report_path):
    """
    Loads a sequence report (JSON lines), returning a dict keyed by genbankAccession,
    for example:
        {
          "OX291655.1": {
             "chrName": "1",
             "gcPercent": 36.0,
             "length": 33474334,
             ...
          },
          "OX291659.1": {
             "chrName": "2",
             "gcPercent": 37.0,
             ...
          },
          ...
        }
    """
    if report_path is None:
        return {}

    seq_report_map = {}
    with open(report_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # The key we match to FASTA records is often the 'genbankAccession'
            # or whichever field you see in record.id
            genbank_acc = data["genbankAccession"]
            seq_report_map[genbank_acc] = data
    return seq_report_map


def parse_genbank_features(gbff_path):
    """
    Parse a .gbff file capturing all relevant features (gene, CDS, mRNA, etc.).
    We store them keyed by the record ID, returning:
       features_map[rec_id] = {
         "description": <record.description>,
         "features": [
            (start, end, feature_type, qualifiers_dict),
            ...
         ]
       }
    """
    if gbff_path is None:
        return {}

    features_map = {}
    for rec in SeqIO.parse(gbff_path, "genbank"):
        rec_id = rec.id
        all_features = []
        for feature in rec.features:
            # Convert Biopython FeatureLocation to integer start/end
            start = int(feature.location.start)
            end   = int(feature.location.end)
            ftype = feature.type
            quals = dict(feature.qualifiers)  # Convert to a normal dict if needed
            # Store each feature's boundaries, type, and qualifiers
            all_features.append((start, end, ftype, quals))

        features_map[rec_id] = {
            "description": rec.description,
            "features": all_features
        }

    return features_map


def chunk_sequence(seq_str, chunk_size=5000, overlap=1000):
    """
    Splits seq_str into overlapping chunks of length chunk_size,
    with the specified overlap between consecutive chunks.
    Returns a list of (chunk_seq, start, end).
    """
    chunks = []
    start_pos = 0
    seq_len = len(seq_str)
    while start_pos < seq_len:
        end_pos = min(start_pos + chunk_size, seq_len)
        chunk_seq = seq_str[start_pos:end_pos]
        chunks.append((chunk_seq, start_pos, end_pos))
        if end_pos == seq_len:
            break
        start_pos += (chunk_size - overlap)
    return chunks


def load_reference_genomes_with_chunks(genomes_info):
    all_chunks = []

    for genome in genomes_info:
        genome_path = genome["genome_path"]
        genome_id   = genome["genome_id"]
        seq_report_map = load_sequence_report(genome.get("sequence_report_path"))
        gbff_map       = parse_genbank_features(genome.get("gbff_path"))

        # Parse FASTA
        for record in SeqIO.parse(genome_path, "fasta"):
            chr_id = record.id
            seq_str = str(record.seq).upper()
            # For demonstration, length was restrict to first 50kb
            # mask this line if you want to use the whole sequence            
            seq_str = seq_str[:50000]

            # Get sequence report data if available
            sr_info = seq_report_map.get(chr_id, {})
            chr_name = sr_info.get("chrName", chr_id)

            # Retrieve the list of all features from the .gbff if it exists
            # e.g. 'features': [(start, end, ftype, quals), ...]
            gbff_info = gbff_map.get(chr_id, {"features": [], "description": ""})
            all_features_for_contig = gbff_info["features"]

            # Now chunk the sequence
            chunks = chunk_sequence(seq_str, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
            for (chunk_seq, start, end) in chunks:
                # Filter only those features that overlap [start, end]
                overlap_features = []
                for (f_start, f_end, ftype, quals) in all_features_for_contig:
                    # Check for overlap
                    if f_end < start or f_start > end:
                        continue
                    overlap_features.append({
                        "feature_start": f_start,
                        "feature_end":   f_end,
                        "type":          ftype,
                        "qualifiers":    quals
                    })

                # Build chunk info
                all_chunks.append({
                    'genome_id': genome_id,
                    'chr_id': chr_id,
                    'chr_name': chr_name,
                    'chunk_seq': chunk_seq,
                    'start': start,
                    'end': end,
                    'extra': {
                        # store the entire gbff description if you want
                        'description': gbff_info["description"],
                        # store only the overlapping features
                        'overlapping_features': overlap_features
                    }
                })

            # If you only want the first record, break here
            break

    return all_chunks


# ----------------------------
# 3. Load and Parse Assembly-Level JSONL Metadata
# ----------------------------

def load_single_assembly_metadata(jsonl_path):
    """
    Load assembly-level metadata (like total_length, n50, etc.) from JSONL.
    Adjust keys as needed. We simply read the first line here.
    """
    if not jsonl_path:
        return {}
    with open(jsonl_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            # Adjust the keys based on actual JSON structure
            return {
                'total_length': entry.get('total_length', 0),
                'num_contigs': entry.get('num_contigs', 0),
                'n50': entry.get('n50', 0),
                'gc_content': entry.get('gc_content', 0.0),
                'busco_complete': entry.get('busco_complete', 0),
            }
    return {}

def load_all_assembly_metadata(genomes_info):
    """
    For each genome in `genomes_info`, load the assembly-level metadata.
    Returns a dict keyed by genome_id -> metadata dict.
    """
    meta_dict = {}
    for genome in genomes_info:
        genome_id      = genome["genome_id"]
        metadata_path  = genome.get("metadata_path", None)
        meta           = load_single_assembly_metadata(metadata_path)
        meta_dict[genome_id] = meta
    return meta_dict


# ----------------------------
# 4. Example usage: Load chunks
# ----------------------------

chunk_records = load_reference_genomes_with_chunks(genomes_info)
print(f"Loaded and chunked {len(chunk_records)} segments total.")

# Load assembly-level metadata
assembly_metadata_dict = load_all_assembly_metadata(genomes_info)
print(f"Loaded metadata for {len(assembly_metadata_dict)} genomes.")

# ----------------------------
# 5. Initialize Model + Tokenizer
# ----------------------------

MODEL_NAME = "bert-base-uncased"   # or a domain-specific model if available
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

# ----------------------------
# 6. K-mer generation + encoding
# ----------------------------

def generate_kmers(sequence, k):
    """ Generate k-mers from the given sequence. """
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

@torch.no_grad()
def encode_kmers(kmers, tokenizer, model, max_len=512):
    """
    Encode a list of k-mer strings into embeddings using the specified model.
    Returns a tensor of shape (num_kmers, hidden_dim).
    """
    if not kmers:
        # if there's no k-mer (e.g., short sequence), return a zero vector
        return torch.zeros((1, model.config.hidden_size), dtype=torch.float32)
    inputs = tokenizer(kmers, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    outputs = model(**inputs)
    # Use [CLS] token embedding
    cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (num_kmers, hidden_dim)
    return cls_embeddings

# ----------------------------
# 7. Build Embeddings for Each Chunk, Incorporating Assembly Metadata
# ----------------------------

# Gather numeric metadata for each chunk. We'll use the assembly-level metadata only,
# but you could also incorporate chromosome-level data from `chunk_info['extra']`.
metadata_arrays = []
for chunk_info in chunk_records:
    meta = assembly_metadata_dict.get(chunk_info['genome_id'], {})
    features = [
        meta.get('total_length', 0),
        meta.get('num_contigs', 0),
        meta.get('n50', 0),
        meta.get('gc_content', 0.0),
        meta.get('busco_complete', 0),
    ]
    metadata_arrays.append(features)

# Normalize these across all chunks
scaler = MinMaxScaler()
metadata_arrays = scaler.fit_transform(metadata_arrays)  # shape: (num_chunks, num_features)
metadata_arrays = torch.tensor(metadata_arrays, dtype=torch.float32)

chunk_embeddings = []
for i, chunk_info in enumerate(chunk_records):
    seq = chunk_info['chunk_seq']
    kmers = generate_kmers(seq, KMER_SIZE)
    kmer_embs = encode_kmers(kmers, tokenizer, model)
    # Mean-pool across the k-mer embeddings for the chunk representation
    chunk_emb = kmer_embs.mean(dim=0)  # shape: (hidden_dim,)
    meta_vec  = metadata_arrays[i]     # shape: (num_features,)

    # Concatenate chunk_emb with the metadata
    chunk_emb_with_meta = torch.cat((chunk_emb, meta_vec), dim=0)
    chunk_embeddings.append(chunk_emb_with_meta)

chunk_embeddings = torch.stack(chunk_embeddings)  # (num_chunks, hidden_dim + num_features)
print("Built chunk embeddings:", chunk_embeddings.shape)

# ----------------------------
# 8. Example Query
# ----------------------------

query = "ATGGGAGAGAACAGAGTTGGAAGTTTTGAACTGAAACGAACAAATACGATCTT"
query_kmers = generate_kmers(query.upper(), KMER_SIZE)
query_embs = encode_kmers(query_kmers, tokenizer, model)
query_emb  = query_embs.mean(dim=0)

# Because chunk embeddings are (hidden_dim + num_meta_features),
# we zero-pad the query to match that dimension:
num_meta_dims = chunk_embeddings.shape[1] - query_emb.shape[0]
query_emb_with_meta = torch.cat((query_emb, torch.zeros(num_meta_dims)), dim=0)

# ----------------------------
# 9. Simple External DB / RAG-like
# ----------------------------

known_genes = {
    "AT1G01010": "ATGGGAGAGAACAGAGTTGGAAGTTTTGAACTGAAACGAACAAATACGATCTT",
    "motif1":    "GATCGATCGA",
}

def retrieve_information(query_str, db):
    retrieved_facts = []
    for key, val in db.items():
        if key in query_str or query_str in key or query_str in val:
            retrieved_facts.append(f"Match found for key='{key}': {val}")
    return list(set(retrieved_facts))

retrieved_info = retrieve_information(query, known_genes)
print("Retrieved external info:", retrieved_info)

# ----------------------------
# 10. Compute Similarity + Softmax
# ----------------------------

scores = torch.matmul(chunk_embeddings, query_emb_with_meta)  # (num_chunks,)
attention_weights = torch.softmax(scores, dim=0)              # (num_chunks,)

# ----------------------------
# 11. Generate Final Annotation
# ----------------------------

def generate_annotation(query_emb, chunk_embeddings, attention_weights, retrieved_info, threshold=0.5):
    # Weighted sum of chunk embeddings
    combined_repr = torch.sum(attention_weights.unsqueeze(1) * chunk_embeddings, dim=0)
    hidden_dim = query_emb.shape[0]
    # Dot-product similarity in embedding subspace
    similarity_score = torch.dot(query_emb, combined_repr[:hidden_dim]) / (
        torch.norm(query_emb) * torch.norm(combined_repr[:hidden_dim])
    )
    score_val = similarity_score.item()

    if score_val > threshold:
        confidence_text = "High match confidence"
    else:
        confidence_text = "Low to moderate match confidence"

    annotation = (
        "Predicted Functional Annotation:\n"
        f"- Similarity Score: {score_val:.4f} ({confidence_text})\n"
        f"- Potential regulatory/coding sequence related to query.\n"
        f"- Integrated External Knowledge: {retrieved_info}\n"
    )
    return annotation

annotation_output = generate_annotation(query_emb, chunk_embeddings, attention_weights, retrieved_info)
print("\n=== Final Annotation ===")
print(annotation_output)

# ----------------------------
# 12. Inspect Top Chunks
# ----------------------------

topN = 3
sorted_scores = sorted(enumerate(scores.tolist()), key=lambda x: x[1], reverse=True)
print(f"\n=== Top {topN} Chunk Matches ===")
for rank, (idx, scr) in enumerate(sorted_scores[:topN], start=1):
    chunk_info = chunk_records[idx]
    att_weight = attention_weights[idx].item()
    print(f"Rank {rank}: Score={scr:.4f}, Attention={att_weight:.4f}")
    print(f"  Genome={chunk_info['genome_id']}, ChrID={chunk_info['chr_id']}, "
          f"  ChrName={chunk_info['chr_name']}, Start={chunk_info['start']}, End={chunk_info['end']}")
    print(f"  Extra={chunk_info['extra']}")

