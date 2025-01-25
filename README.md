# BreedLLM: A Unified Large Language Model Framework for Accelerated Crop Improvement Manuscript Proof-of-Concepts

This repository contains Python code implementing proof-of-concepts from the BreedLLM manuscript. It demonstrates two main workflows:

- Script 1: PanGenomic-LM  
- Script 2: LangChain Integration with R-based BLUP Analysis

---

## Table of Contents

1. Overview
2. Requirements
3. Installation
4. Usage
    - Script 1: PanGenomic-LM
    - Script 2: LangChain & R-based BLUP Analysis
5. License
6. Acknowledgments

---

## Overview

### Script 1: PanGenomic-LM

This script demonstrates a workflow for processing multiple genomes by representing them as text and then applying transformer-based techniques. 
The key steps include:

- Pangenome Representation:  
  Multiple reference genomes are represented as text (FASTA-like format).

- Embedding Generation:  
  A transformer-based model encodes these representations into embeddings.

- Query & Scoring:  
  A query is defined and relevance scores (using attention or similarity metrics) are computed between the query and each genome embedding.

- Retrieval-Augmented Generation (RAG): 
  External annotations or metadata are integrated by combining embedding-based retrieval with external databases.

- Output Generation:  
  The script outputs final annotations or predictions, highlighting which genome segments best match the query.

### Script 2: LangChain & R-based BLUP Analysis

This script demonstrates an integration between Python, LangChain, and an R-based analysis for Best Linear Unbiased Prediction (BLUP). 
  The workflow includes:

- Toy Data Creation:
    - The script generates a small CSV file (data.csv) with toy data if it doesn't already exist.

- R Script Setup:
  - It checks for the presence of an R script (blup_analysis.R) in the same directory and creates it if needed.
  - The R script uses the lme4 package to perform BLUP analysis.
  - The script ensures that blup_analysis.R is executable on Unix-like systems.

- LangChain Setup:
  - The script uses LangChain to set up an agent that triggers the execution of blup_analysis.R.
  - The LangChain integration requires setting your OpenAI API key or having a locally running LLM instance.

#### Requirements
Before running the scripts, ensure you have the following:

- Python 3.10.11 (Recommended; though Python 3.7+ may work with potential modifications)
- R installed on your system (preferably version 4.0+)
  - The R package lme4 must be installed for the BLUP analysis
- OpenAI API Key (if you intend to use the LangChain integration with OpenAI). Alternatively, configure a locally running LLM instance.

- Python Dependencies
  All required Python packages are specified in requirements.txt.
  Run the following to install them:
  ```bash
  pip install -r requirements.txt
  ```
 Alternatively, you can manually install the key libraries for each script:

    For Script 1 (PanGenomic-LM):
        biopython==1.83
        torch==2.0.1
        transformers==4.34.0
        numpy==1.26.2
        scikit-learn==1.3.1
        (The json module is part of the standard library.)

    For Script 2 (LangChain & R-based BLUP):
        langchain==0.1.0
        openai==1.2.0
        numpy==1.26.2
        scikit-learn==1.3.1

### Installation

Below are step-by-step instructions to set up the environment for both scripts.
- Clone This Repository
    git clone https://github.com/sikiru-atanda/BreedLLM_manuscript
    cd BreedLLM
- Create and Activate a Python Virtual Environment (Recommended)
    python -m venv pangenomic_env
    source pangenomic_env/bin/activate   # For Linux/MacOS
    # On Windows:
    # pangenomic_env\Scripts\activate
- Install Dependencies
    pip install --upgrade pip
    pip install -r requirements.txt
- Set Up R for BreedLLM Guided BLUP Analysis (Script 2)
  In your R console:
    install.packages("lme4")
NOTE: Ensure you have R accessible in your system’s PATH so that Python can call it.
- Configure OpenAI API Key
  If you intend to use OpenAI for LangChain:
    export OPENAI_API_KEY="your-api-key-here"

### Usage
Script 1: PanGenomic-LM

- Prepare Genome Data:
    - Ensure your genome sequences are available in a text-based format (e.g., FASTA-like).
    - Adjust file paths in breed_llm_pangenomic.py if necessary

- Configure the Script:
    - Edit chunk size, overlap, model name, or other parameters in breed_llm_pangenomic.py if desired.

- Run the Script:
	  python breed_llm_pangenomic.py
	
- What the script does:
    - Parse genome sequences using Biopython.
    - Generates embeddings for k-mers using the specified transformer model.
    - Computes similarity scores or attention between the query embedding and genome chunk embeddings.
    - Integrates external metadata (RAG-like approach)
    - Outputs annotations or predictions indicating the most relevant genome segments.
	
Script 2: LangChain & R-based BLUP Analysis

- Ensure Toy Data & R Script Setup:
   - breed_llm_guided_blup_analysis.py generates data.csv (toy dataset) if none is found.
   - It checks for blup_analysis.R and creates it if missing.
   - On Unix-like systems, the script attempts to run "blup_analysis.R" to make the R script executable:   
       chmod +x blup_analysis.R
   - Configure LangChain Credentials:
      - Set your OPENAI_API_KEY environment variable, or configure a local LLM in the script

 - Run the Script:
	python breed_llm_guided_blup_analysis.py
	
	- What the script does:
    - Create and verify toy data (data.csv).
    - Ensure the presence of blup_analysis.R.
    - Uses LangChain to initialize a custom agent that calls the R script.
    - The R script runs a simple BLUP analysis using the lme4 package.
    - Results are returned to Python for final output and logging.
	
### Acknowledgments
- BreedLLM Manuscript: These proof-of-concepts are based on ideas from “BreedLLM: 
                        A Unified Large Language Model Framework for Accelerated Crop Improvement.”
- Contributors: Thanks to all collaborators for feedback, insights, and testing.
- Libraries & Tools: We gratefully acknowledge the developers of Biopython, PyTorch, Transformers, LangChain, and the R community (especially the lme4 authors).

Contact: For questions, please open an issue or contact the corresponding author of the manuscript.

Enjoy developing your version of BreedLLM for advanced genomic and crop improvement research!