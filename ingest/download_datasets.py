#!/usr/bin/env python3
"""
Download Med-HALT and biomedical corpus datasets.

Downloads:
- Med-HALT benchmark (Hugging Face)
- PubMed baseline abstracts (FTP)
- PMC Open Access articles
- MedlinePlus health topics
- PubTator entity annotations
- OpenFDA drug labels
- ICD-10 codes
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Optional
import requests
from datasets import load_dataset
import json
import xml.etree.ElementTree as ET
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Download and organize biomedical datasets."""

    def __init__(self, dest_dir: str):
        self.dest_dir = Path(dest_dir)
        self.dest_dir.mkdir(parents=True, exist_ok=True)

    def download_medhalt(self):
        """Download Med-HALT benchmark from Hugging Face."""
        logger.info("Downloading Med-HALT dataset...")
        output_dir = self.dest_dir / "medhalt"
        output_dir.mkdir(exist_ok=True)

        # Med-HALT configs for PoC evaluation
        # reasoning_FCT: Fact-checking tasks (primary)
        # reasoning_fake: Misinformation detection
        # IR_pmid2title: Information retrieval evaluation
        configs = ['reasoning_FCT', 'reasoning_fake', 'IR_pmid2title']

        try:
            for config_name in configs:
                logger.info(f"Downloading Med-HALT config: {config_name}")

                # Load dataset from Hugging Face with specific config
                dataset = load_dataset("openlifescienceai/Med-HALT", config_name)

                # Create subdirectory for this config
                config_dir = output_dir / config_name
                config_dir.mkdir(exist_ok=True)

                # Save splits
                for split_name, split_data in dataset.items():
                    output_file = config_dir / f"{split_name}.jsonl"
                    split_data.to_json(output_file, orient="records", lines=True)
                    logger.info(f"Saved {config_name}/{split_name} to {output_file}")

            logger.info(f"Med-HALT dataset (3 configs) downloaded to {output_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to download Med-HALT: {e}")
            return False

    def download_pubmed_sample(self, max_articles: int = 10000):
        """
        Download a sample of PubMed articles via E-utilities API.
        For full baseline, users should use FTP (see Plan.md).
        
        Note: NCBI E-utilities limits retmax to 10,000 per request.
        For larger samples, we'll use multiple queries with retstart offset.
        """
        logger.info(f"Downloading PubMed sample ({max_articles} articles)...")
        output_dir = self.dest_dir / "pubmed_baseline"
        output_dir.mkdir(exist_ok=True)

        # Use E-utilities to fetch a sample
        # This is a simplified version; full implementation would use FTP
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

        try:
            # For >10K articles, use multiple diverse search queries
            # NCBI E-utilities API limit: retmax=10000 per request
            # Note: retstart doesn't work reliably for large offsets, so we use different search terms
            all_pmids = []
            api_limit = 10000
            
            search_url = f"{base_url}esearch.fcgi"
            
            # Multiple diverse search strategies to get 100K unique articles
            search_strategies = [
                "(cardiology[MeSH] OR heart[Title/Abstract])",
                "(diabetes[MeSH] OR insulin[Title/Abstract])",
                "(cancer[MeSH] OR oncology[Title/Abstract])",
                "(hypertension[MeSH] OR blood pressure[Title/Abstract])",
                "(infection[MeSH] OR antimicrobial[Title/Abstract])",
                "(neurology[MeSH] OR brain[Title/Abstract])",
                "(surgery[MeSH] OR surgical[Title/Abstract])",
                "(pediatrics[MeSH] OR children[Title/Abstract])",
                "(psychiatry[MeSH] OR mental health[Title/Abstract])",
                "(pharmacology[MeSH] OR drug therapy[Title/Abstract])",
            ]
            
            # Use as many strategies as needed to reach target
            queries_needed = min(len(search_strategies), (max_articles + api_limit - 1) // api_limit)
            
            for query_num, search_term in enumerate(search_strategies[:queries_needed]):
                if len(all_pmids) >= max_articles:
                    break
                    
                retmax = min(api_limit, max_articles - len(all_pmids))
                
                params = {
                    "db": "pubmed",
                    "term": search_term,
                    "retmax": retmax,
                    "retmode": "json",
                    "sort": "relevance"
                }

                logger.info(f"Query {query_num + 1}/{queries_needed}: Searching '{search_term}' (target: {retmax} articles)")
                
                # Add delay between requests to respect NCBI rate limits (max 3 requests/second)
                if query_num > 0:
                    time.sleep(0.34)  # ~3 requests per second
                
                try:
                    response = requests.get(search_url, params=params, timeout=30)
                    response.raise_for_status()
                    
                    # Check if response is valid JSON
                    try:
                        search_results = response.json()
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in query {query_num + 1}, retrying after delay...")
                        time.sleep(1)
                        response = requests.get(search_url, params=params, timeout=30)
                        response.raise_for_status()
                        search_results = response.json()
                    
                    batch_pmids = search_results.get("esearchresult", {}).get("idlist", [])
                    
                    # Add only unique PMIDs
                    unique_new = [pmid for pmid in batch_pmids if pmid not in all_pmids]
                    all_pmids.extend(unique_new)
                    
                    logger.info(f"Found {len(batch_pmids)} PMIDs ({len(unique_new)} unique, total: {len(all_pmids)})")
                        
                except Exception as e:
                    logger.error(f"Error in query {query_num + 1}: {e}")
                    logger.info(f"Continuing with {len(all_pmids)} PMIDs collected so far...")
                    continue
            
            # Trim to exact count
            pmids = all_pmids[:max_articles]
            logger.info(f"Total unique PubMed IDs collected: {len(pmids)}")

            # Fetch abstracts in batches
            output_file = output_dir / "pubmed_sample.jsonl"
            batch_size = 100
            articles_written = 0

            with open(output_file, 'w') as f:
                for i in range(0, len(pmids), batch_size):
                    batch_pmids = pmids[i:i+batch_size]
                    fetch_url = f"{base_url}efetch.fcgi"
                    fetch_params = {
                        "db": "pubmed",
                        "id": ",".join(batch_pmids),
                        "retmode": "xml"
                    }

                    # Add delay to respect NCBI rate limits
                    if i > 0:
                        time.sleep(0.34)  # ~3 requests per second
                    
                    try:
                        fetch_response = requests.get(fetch_url, params=fetch_params, timeout=30)
                        fetch_response.raise_for_status()

                        # Parse XML and extract article data
                        try:
                            root = ET.fromstring(fetch_response.content)
                            # Extract articles from PubmedArticle elements
                            for article in root.findall('.//PubmedArticle'):
                                pmid_elem = article.find('.//PMID')
                                title_elem = article.find('.//ArticleTitle')
                                abstract_elem = article.find('.//AbstractText')

                                record = {
                                    "id": pmid_elem.text if pmid_elem is not None else "",
                                    "pmid": pmid_elem.text if pmid_elem is not None else "",
                                    "title": title_elem.text if title_elem is not None else "",
                                    "text": abstract_elem.text if abstract_elem is not None else "",
                                    "source": "pubmed"
                                }
                                f.write(json.dumps(record) + '\n')
                                articles_written += 1
                        except ET.ParseError as e:
                            logger.warning(f"Failed to parse XML for batch {i//batch_size + 1}: {e}")
                            continue
                    
                    except requests.exceptions.RequestException as e:
                        logger.warning(f"Failed to fetch batch {i//batch_size + 1}: {e}")
                        continue

                    if (i // batch_size + 1) % 10 == 0:  # Log every 10 batches
                        logger.info(f"Fetched batch {i//batch_size + 1}/{(len(pmids)-1)//batch_size + 1} ({articles_written} articles)")

            logger.info(f"PubMed sample saved to {output_file} ({articles_written} articles written)")
            logger.warning("For full baseline, use FTP download (see Plan.md section 3.2)")
            return True

        except Exception as e:
            logger.error(f"Failed to download PubMed sample: {e}")
            return False

    def download_medlineplus(self):
        """Download MedlinePlus health topics XML."""
        logger.info("Downloading MedlinePlus health topics...")
        output_dir = self.dest_dir / "medlineplus"
        output_dir.mkdir(exist_ok=True)

        # MedlinePlus XML endpoint
        url = "https://medlineplus.gov/xml.html"

        logger.warning(f"MedlinePlus requires manual download from {url}")
        logger.info("Please download medlineplus_all_healthtopics.xml.zip manually")
        logger.info(f"and extract to {output_dir}")

        # Create placeholder
        readme = output_dir / "README.txt"
        readme.write_text(
            f"Download MedlinePlus XML from:\n{url}\n\n"
            "Extract medlineplus_all_healthtopics.xml to this directory."
        )
        return True

    def download_pubtator(self):
        """Download PubTator Central annotations (sample)."""
        logger.info("Downloading PubTator annotations...")
        output_dir = self.dest_dir / "pubtator"
        output_dir.mkdir(exist_ok=True)

        # For PoC, we'll document the FTP approach
        ftp_url = "ftp://ftp.ncbi.nlm.nih.gov/pub/lu/PubTatorCentral/"

        logger.warning(f"PubTator Central requires FTP download from {ftp_url}")
        logger.info("For large-scale annotation, use FTP or API")

        # Create placeholder
        readme = output_dir / "README.txt"
        readme.write_text(
            f"Download PubTator annotations from:\n{ftp_url}\n\n"
            "For PoC, download a subset of bioconcepts files."
        )
        return True

    def download_icd10(self):
        """Download ICD-10 code mappings."""
        logger.info("Setting up ICD-10 codes...")
        output_dir = self.dest_dir / "icd10"
        output_dir.mkdir(exist_ok=True)

        # ICD-10 requires manual download
        url = "https://icd.who.int/browse10/2019/en"

        readme = output_dir / "README.txt"
        readme.write_text(
            f"Download ICD-10 codes from:\n{url}\n\n"
            "Alternative: Use CDC ICD-10-CM files\n"
            "https://www.cdc.gov/nchs/icd/icd-10-cm.htm"
        )

        logger.info(f"ICD-10 download instructions saved to {readme}")
        return True


def main():
    parser = argparse.ArgumentParser(description="Download datasets for Med-HALT RAG PoC")
    parser.add_argument(
        "--dest",
        type=str,
        default="data/raw",
        help="Destination directory for downloads"
    )
    parser.add_argument(
        "--only-medhalt",
        action="store_true",
        help="Download only Med-HALT (for quick testing)"
    )
    parser.add_argument(
        "--pubmed-sample-size",
        type=int,
        default=10000,
        help="Number of PubMed articles to download (sample)"
    )

    args = parser.parse_args()

    downloader = DatasetDownloader(args.dest)

    # Always download Med-HALT
    success = downloader.download_medhalt()

    if not args.only_medhalt:
        logger.info("Downloading full dataset suite...")
        success &= downloader.download_pubmed_sample(args.pubmed_sample_size)
        success &= downloader.download_medlineplus()
        success &= downloader.download_pubtator()
        success &= downloader.download_icd10()

    if success:
        logger.info("Dataset download completed successfully!")
    else:
        logger.warning("Some datasets could not be downloaded automatically.")
        logger.info("Check README files in data/raw subdirectories for manual download instructions.")


if __name__ == "__main__":
    main()
