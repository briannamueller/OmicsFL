"""GDC (Genomic Data Commons) API download backend.

Downloads STAR-Counts expression and clinical metadata from the NCI GDC
REST API. Uses only stdlib urllib.
"""
from __future__ import annotations

import io
import json
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from omicsfl.download.registry import gdc_project_id, validate_cohorts

_GDC_API = "https://api.gdc.cancer.gov"
_BATCH_SIZE = 30
_MAX_RETRIES = 3


def _gdc_post(endpoint, payload, timeout=120):
    url = f"{_GDC_API}/{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _gdc_get(endpoint, params, timeout=60):
    query = "&".join(
        f"{k}={urllib.request.quote(str(v))}" for k, v in params.items()
    )
    url = f"{_GDC_API}/{endpoint}?{query}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _query_expression_files(project_id):
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {
                "field": "cases.project.project_id",
                "value": [project_id],
            }},
            {"op": "in", "content": {
                "field": "data_type",
                "value": ["Gene Expression Quantification"],
            }},
            {"op": "in", "content": {
                "field": "analysis.workflow_type",
                "value": ["STAR - Counts"],
            }},
        ],
    }

    files = []
    page_from = 0
    page_size = 500

    while True:
        params = {
            "filters": json.dumps(filters),
            "fields": ("file_id,file_name,"
                       "cases.submitter_id,"
                       "cases.samples.submitter_id,"
                       "cases.samples.sample_type"),
            "size": str(page_size),
            "from": str(page_from),
            "format": "json",
        }
        resp = _gdc_get("files", params)
        hits = resp["data"]["hits"]
        if not hits:
            break

        for hit in hits:
            case = hit["cases"][0]
            sample = case["samples"][0]
            files.append({
                "file_id": hit["file_id"],
                "file_name": hit["file_name"],
                "case_id": case["submitter_id"],
                "sample_barcode": sample["submitter_id"],
                "sample_type": sample["sample_type"],
            })

        total = resp["data"]["pagination"]["total"]
        page_from += page_size
        if page_from >= total:
            break

    return files


def _download_batch(file_ids, output_dir, batch_num, total_batches):
    print(f"    Batch {batch_num}/{total_batches} "
          f"({len(file_ids)} files) ...", end=" ", flush=True)

    payload = json.dumps({"ids": file_ids}).encode("utf-8")

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(
                f"{_GDC_API}/data",
                data=payload,
                headers={"Content-Type": "application/json"},
            )

            extracted = []
            with urllib.request.urlopen(req, timeout=600) as resp:
                raw = resp.read()

                if len(file_ids) == 1:
                    dest = output_dir / file_ids[0]
                    dest.mkdir(parents=True, exist_ok=True)
                    out_path = dest / f"{file_ids[0]}.tsv"
                    with open(out_path, "wb") as f:
                        f.write(raw)
                    extracted.append(out_path)
                else:
                    tar_data = io.BytesIO(raw)
                    with tarfile.open(fileobj=tar_data, mode="r:gz") as tar:
                        tar.extractall(path=str(output_dir))
                        extracted = [
                            output_dir / m.name
                            for m in tar.getmembers() if m.isfile()
                        ]

            print(f"done ({len(extracted)} files)")
            return extracted

        except (EOFError, tarfile.ReadError, urllib.error.URLError,
                ConnectionError, TimeoutError, OSError) as exc:
            if attempt < _MAX_RETRIES:
                wait = 10 * attempt
                print(f"\n      Retry {attempt} ({exc}), "
                      f"waiting {wait}s ...", end=" ", flush=True)
                time.sleep(wait)
            else:
                print(f"\n      FAILED after {_MAX_RETRIES} attempts")
                raise


def _build_counts_matrix(file_infos, download_dir):
    print("  Building counts matrix ...")
    counts_dict = {}
    gene_names = None

    for i, info in enumerate(file_infos):
        fid = info["file_id"]
        fname = info["file_name"]
        barcode = info["sample_barcode"]

        tsv_path = download_dir / fid / fname
        if not tsv_path.exists():
            fid_dir = download_dir / fid
            if fid_dir.exists():
                tsvs = list(fid_dir.glob("*.tsv"))
                if tsvs:
                    tsv_path = tsvs[0]
                else:
                    continue
            else:
                continue

        df = pd.read_csv(tsv_path, sep="\t", comment="#",
                         index_col=0, header=0)
        if "unstranded" in df.columns:
            counts = df["unstranded"]
        else:
            counts = df.iloc[:, 0]

        counts = counts[~counts.index.str.startswith(("N_", "__"))]

        if gene_names is None:
            gene_names = counts.index.tolist()

        counts_dict[barcode] = counts.values

        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(file_infos)} files processed")

    matrix = pd.DataFrame(counts_dict, index=gene_names).T
    print(f"  Matrix: {matrix.shape[0]} samples x {matrix.shape[1]} genes")
    return matrix


def _download_clinical(project_id):
    filters = {
        "op": "in",
        "content": {
            "field": "project.project_id",
            "value": [project_id],
        },
    }
    fields = [
        "submitter_id",
        "demographic.gender",
        "diagnoses.ajcc_pathologic_stage",
        "diagnoses.tumor_stage",
    ]

    cases = []
    page_from = 0
    page_size = 500

    while True:
        params = {
            "filters": json.dumps(filters),
            "fields": ",".join(fields),
            "size": str(page_size),
            "from": str(page_from),
            "format": "json",
        }
        resp = _gdc_get("cases", params)
        hits = resp["data"]["hits"]
        if not hits:
            break

        for hit in hits:
            case_id = hit.get("submitter_id", "")
            gender = hit.get("demographic", {}).get("gender", "")
            diagnoses = hit.get("diagnoses", [{}])
            stage = ""
            if diagnoses:
                stage = diagnoses[0].get("ajcc_pathologic_stage", "")
                if not stage:
                    stage = diagnoses[0].get("tumor_stage", "")

            cases.append({
                "case_id": case_id,
                "gender": gender,
                "stage": stage,
            })

        total = resp["data"]["pagination"]["total"]
        page_from += page_size
        if page_from >= total:
            break

    return pd.DataFrame(cases)


def download(
    cohorts: list[str],
    output_dir: str | Path = "./omicsfl_data/gdc",
) -> dict[str, Path]:
    """Download TCGA expression counts and clinical data from GDC.

    Returns dict mapping cohort -> {"counts": Path, "clinical": Path}.
    """
    cohorts = validate_cohorts(cohorts)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[omicsfl.download.gdc] Downloading from GDC API")
    print(f"  Cohorts: {', '.join(cohorts)}")
    print(f"  Output: {output_dir}")
    print()

    results = {}

    for cohort in cohorts:
        project_id = gdc_project_id(cohort)
        cohort_dir = output_dir / cohort
        cohort_dir.mkdir(parents=True, exist_ok=True)

        counts_file = cohort_dir / "counts_matrix.parquet"
        clinical_file = cohort_dir / "clinical.csv"

        print(f"[{cohort}] {project_id}")

        # Expression counts
        if counts_file.exists():
            print(f"  Counts already cached: {counts_file.name}")
        else:
            print(f"  Querying expression files ...")
            file_infos = _query_expression_files(project_id)
            print(f"  Found {len(file_infos)} files")

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                total_batches = (
                    (len(file_infos) + _BATCH_SIZE - 1) // _BATCH_SIZE
                )

                for batch_idx in range(total_batches):
                    start = batch_idx * _BATCH_SIZE
                    end = min(start + _BATCH_SIZE, len(file_infos))
                    batch_ids = [f["file_id"] for f in file_infos[start:end]]
                    _download_batch(
                        batch_ids, tmpdir, batch_idx + 1, total_batches,
                    )

                matrix = _build_counts_matrix(file_infos, tmpdir)

            matrix.to_parquet(counts_file)
            size_mb = counts_file.stat().st_size / (1024 * 1024)
            print(f"  Saved: {counts_file.name} ({size_mb:.1f} MB)")

        # Clinical data
        if clinical_file.exists():
            print(f"  Clinical already cached: {clinical_file.name}")
        else:
            print(f"  Downloading clinical data ...")
            clinical = _download_clinical(project_id)
            clinical.to_csv(clinical_file, index=False)
            print(f"  Saved: {clinical_file.name} ({len(clinical)} records)")

        results[cohort] = {
            "counts": counts_file,
            "clinical": clinical_file,
        }
        print()

    return results
