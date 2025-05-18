# Artifacts for paper submission: Semantic Reasoning for Mining Performance Improvement Commits at Scale via LLMs

## Abstract

Performance bugs degrade software reliability, causing latency, wasted resources, and system failures. Mining software repositories is effective in identifying performance bugs, regressions, and improvements, supporting various reliability-focused tasks. However, existing mining methods typically rely on manual data curation, heuristic-based rules, keyword filtering, or traditional NLP techniques. These approaches often lack semantic accuracy, scalability, and generalizability, limiting their ability to systematically identify meaningful performance improvements at scale.

To overcome these challenges, we propose PerfMiner, a transformer-based framework designed to mine and classify performance improvement commits from large-scale open-source repositories. PerfMiner leverages PerfAnnotator-mini, a compact transformer model trained via knowledge distillation from a larger, semantically sophisticated large language model (LLM). Our approach achieves classification performance (F1-score=0.78) comparable to significantly larger models but with dramatically lower computational requirements, enabling efficient distributed analysis.

Using PerfMiner, we curate and validate a multi-language dataset (C++, Java, Python) comprising 186.9K commits explicitly labeled as performance improvements and performance-related bug fixes. Empirical evaluations demonstrate that models trained on this large-scale labeled dataset significantly outperform those relying on smaller or keyword-filtered datasets, achieving greater effectiveness in detecting performance issues. By combining semantic reasoning with computational scalability, PerfMiner provides a robust methodological foundation, substantially enhancing software reliability research.

## Environment Setup 

1. Create a virtual environment:
    ```bash
    python3 -m venv perfminer
    source perfminer/bin/activate
    # or using conda:
    # conda create -n myenv python=3.9
    # conda activate perfminer
    ```

2. Install all dependencies:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

3. The above environment is required to replicate the RQ experiments and results.


## For data and results presented in the paper

- PerfAnnotator-mini directory contains link to the model for download and usage instructions for inference.
- For the data and results corresponding to each Research Question (RQ), please refer to the respective subdirectories under the /RQs directory.
- The PerfMiner mining framework tool and the large-scale dataset are currently available upon request. We plan to release them publicly in the near future.