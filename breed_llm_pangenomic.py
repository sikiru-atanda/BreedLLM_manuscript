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

from Bio import SeqIO
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler

# ----------------------------
# 1. Parameters and File Paths
# ----------------------------

# Define k-mer size
KMER_SIZE = 5  # Example k-mer size; adjust based on your requirements

# Define how to chunk each reference sequence
CHUNK_SIZE = 5000  # in this example we use 5kb
OVERLAP = 1000     # Overlap between chunks

# List of reference genomes and their corresponding metadata files
genomes_info = [
    {
        "genome_path": "GCF_000001735.4_TAIR10.1_genomic.fna",
        "metadata_path": "assembly_data_report.jsonl",
        "genome_id": "GCF_000001735.4"  # Replace with actual genome ID
    },
    {
        "genome_path": "GCF_000001735.4_TAIR10.1_genomic_col.fna",
        "metadata_path": "assembly_data_report_col.jsonl",
        "genome_id": "GCF_000002285.3"
    },
    # Add more genomes as needed
]

# ----------------------------
# 2. Load Reference Genomes with Chromosome/Position Chunking
# ----------------------------

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
    """
    For each genome in genomes_info:
      - Parse the FASTA file (assuming the first record if multi-record).
      - Chunk the sequence using chunk_sequence().
      - Return a list of (genome_id, chr_id, chunk_seq, start, end).
      - chr_id is taken from record.id if available.

    Returns:
        A list of tuples: [
          {
            'genome_id': ...,
            'chr_id': ...,
            'chunk_seq': ...,
            'start': ...,
            'end': ...
          },
          ...
        ]
    """
    chunk_records = []
    for genome in genomes_info:
        genome_path = genome["genome_path"]
        genome_id = genome["genome_id"]
        # In many reference FASTA files, each record might correspond to a chromosome or contig.
        # This example grabs only the FIRST record per file. Adapt as needed.
        for record in SeqIO.parse(genome_path, "fasta"):
            chr_id = record.id
            seq_str = str(record.seq).upper()  # Convert to uppercase
            # For demonstration, length was restrict to first 50kb
            # mask this line if you want to use the whole sequence
            seq_str = seq_str[:50000]  
            # Now chunk this chromosome/contig
            chunks = chunk_sequence(seq_str, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
            for (chunk_seq, start, end) in chunks:
                chunk_records.append({
                    'genome_id': genome_id,
                    'chr_id': chr_id,
                    'chunk_seq': chunk_seq,
                    'start': start,
                    'end': end
                })
            break  # if you only want the first record in each file
    return chunk_records

chunk_records = load_reference_genomes_with_chunks(genomes_info)
print(f"Loaded and chunked {len(chunk_records)} chromosome/contig segments total.")

# ----------------------------
# 3. Load and Parse Assembly Metadata
# ----------------------------

def load_single_assembly_metadata(jsonl_path):
    """
    Load assembly metadata from a single JSONL file.
    Assumes there's exactly 1 JSON line or the first line is used.
    """
    metadata = {}
    with open(jsonl_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            # Adjust the keys based on actual JSON structure
            metadata = {
                'total_length': entry.get('total_length', 0),
                'num_contigs': entry.get('num_contigs', 0),
                'n50': entry.get('n50', 0),
                'gc_content': entry.get('gc_content', 0.0),
                'busco_complete': entry.get('busco_complete', 0),
                # Add more fields as needed
            }
            break  # only read the first JSON line
    return metadata

def load_all_assembly_metadata(genomes_info):
    """
    Loads metadata for each genome in genomes_info.
    Returns a dict keyed by genome_id -> metadata dict.
    """
    meta_dict = {}
    for genome in genomes_info:
        metadata_path = genome["metadata_path"]
        genome_id = genome["genome_id"]
        meta = load_single_assembly_metadata(metadata_path)
        meta_dict[genome_id] = meta
    return meta_dict

assembly_metadata_dict = load_all_assembly_metadata(genomes_info)
print(f"Loaded metadata for {len(assembly_metadata_dict)} genomes.")

# ----------------------------
# 4. Known External Information (RAG-Like)
# ----------------------------

known_genes = {
    "AT1G01010": "ATGGGAGAGAACAGAGTTGGAAGTTTTGAACTGAAACGAACAAATACGATCTT",
    "motif1": "GATCGATCGA",
}

# ----------------------------
# 5. Initialize Model and Tokenizer
# ----------------------------

# Replace with a genomics-specific model if available
MODEL_NAME = "bert-base-uncased"  
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

# ----------------------------
# 6. K-mer Generation
# ----------------------------

def generate_kmers(sequence, k):
    """
    Generate all possible k-mers from the input sequence using a sliding window of size k.
    """
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

# ----------------------------
# 7. Encode K-mers into Embeddings
# ----------------------------

def encode_kmers(kmers, tokenizer, model, max_len=512):
    """
    Encode a list of k-mer strings into embeddings using the specified model.
    Returns a tensor of shape (num_kmers, hidden_dim).
    """
    with torch.no_grad():
        inputs = tokenizer(kmers, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        outputs = model(**inputs)
        # Use [CLS] token embedding as a summary
        # (num_kmers, hidden_dim)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  
    return cls_embeddings

# ----------------------------
# 8. Build Embeddings for Each Chunk, Incorporate Assembly Metadata
# ----------------------------

# First, gather all numeric metadata for scaling
# We'll map each chunk to its genome_id, then retrieve that genome's metadata
metadata_arrays = []
for chunk_info in chunk_records:
    genome_id = chunk_info['genome_id']
    meta = assembly_metadata_dict.get(genome_id, {})
    features = [
        meta.get('total_length', 0),
        meta.get('num_contigs', 0),
        meta.get('n50', 0),
        meta.get('gc_content', 0.0),
        meta.get('busco_complete', 0),
    ]
    metadata_arrays.append(features)

# Normalize across all chunks
scaler = MinMaxScaler()
metadata_arrays = scaler.fit_transform(metadata_arrays)
# shape: (num_chunks, num_features)
metadata_arrays = torch.tensor(metadata_arrays, dtype=torch.float32)  

chunk_embeddings = []
for i, chunk_info in enumerate(chunk_records):
    seq = chunk_info['chunk_seq']
    # Generate k-mers
    kmers = generate_kmers(seq, KMER_SIZE)
    kmer_embs = encode_kmers(kmers, tokenizer, model)
    # Mean-pool k-mer embeddings to get a single vector representing this chunk
    # shape: (hidden_dim,)
    chunk_emb = kmer_embs.mean(dim=0)  
    # Concatenate normalized metadata
    # shape: (num_features,)
    meta_vec = metadata_arrays[i] 
    # shape: (hidden_dim + num_features,)
    chunk_emb_with_meta = torch.cat((chunk_emb, meta_vec), dim=0)  
    chunk_embeddings.append(chunk_emb_with_meta)

# shape: (num_chunks, hidden_dim + num_features)
chunk_embeddings = torch.stack(chunk_embeddings)  
print("Built embeddings for all chunks:", chunk_embeddings.shape)

# ----------------------------
# 9. Query Embedding
# ----------------------------

def retrieve_information(query_str, db):
    """
    Trivial function to find substring matches in known_genes or other db.
    """
    retrieved_facts = []
    for key, val in db.items():
        if key in query_str or query_str in key or query_str in val:
            retrieved_facts.append(f"Match found for key='{key}': {val}")
    return list(set(retrieved_facts))

# Example query
query = "ATGGGAGAGAACAGAGTTGGAAGTTTTGAACTGAAACGAACAAATACGATCTT"
query_kmers = generate_kmers(query.upper(), KMER_SIZE)
query_embs = encode_kmers(query_kmers, tokenizer, model)
query_emb = query_embs.mean(dim=0)  # (hidden_dim,)

# Since the reference embeddings each have extra metadata dims appended,
# we pad the query with zeros for those dims (like the example above).
num_meta_dims = chunk_embeddings.shape[1] - query_emb.shape[0]
query_emb_with_meta = torch.cat((query_emb, torch.zeros(num_meta_dims)), dim=0)

# RAG-like external retrieval
retrieved_info = retrieve_information(query, known_genes)
print("Retrieved external info:", retrieved_info)

# ----------------------------
# 10. Similarity and Attention Weights
# ----------------------------

# We'll do a simple dot-product to get a similarity score, then softmax
# shape: (num_chunks,)
scores = torch.matmul(chunk_embeddings, query_emb_with_meta) 
# shape: (num_chunks,)
attention_weights = torch.softmax(scores, dim=0)  

# ----------------------------
# 11. Generate Final Annotation
# ----------------------------

def generate_annotation(query_emb, chunk_embeddings, attention_weights, retrieved_info, threshold=0.5):
    """
    Aggregates chunk embeddings by attention, compares to query for a final similarity.
    Includes a toy threshold for "high confidence."
    """
    # Weighted sum of chunk embeddings
    combined_repr = torch.sum(attention_weights.unsqueeze(1) * chunk_embeddings, dim=0)
    # For final similarity, compare the portion of combined_repr that corresponds to the original embedding size
    hidden_dim = query_emb.shape[0]
    # Dot-product similarity in the embedding subspace
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
        f"- Potential regulatory region or coding sequence related to the query.\n"
        f"- Integrated External Knowledge: {retrieved_info}\n"
    )
    return annotation

annotation_output = generate_annotation(query_emb, chunk_embeddings, attention_weights, retrieved_info)
print("\n=== Final Annotation ===")
print(annotation_output)

# ----------------------------
# 12. Inspect Top Chunks by Similarity
# ----------------------------
# If you want to see which exact chunks contributed most, you can look at the "scores" or "attention_weights".
# E.g., print the top N matches with their chunk metadata. Here we print top 3:
topN = 3
sorted_scores = sorted(enumerate(scores.tolist()), key=lambda x: x[1], reverse=True)  # (index, score)
print(f"\n=== Top {topN} Chunk Matches ===")
for rank, (idx, scr) in enumerate(sorted_scores[:topN], start=1):
    chunk_info = chunk_records[idx]
    att_weight = attention_weights[idx].item()
    print(f"Rank {rank}: Score={scr:.4f}, Attention={att_weight:.4f}")
    print(f"  Genome={chunk_info['genome_id']}, Chr={chunk_info['chr_id']}, "
          f"Start={chunk_info['start']}, End={chunk_info['end']}")

