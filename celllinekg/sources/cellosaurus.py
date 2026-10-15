from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class CellosaurusEntry:
    name: str = ""
    accession: str = ""
    secondary_accessions: list[str] = field(default_factory=list)
    synonyms: list[str] = field(default_factory=list)
    cross_references: list[dict[str, str]] = field(default_factory=list)
    references: list[dict[str, str]] = field(default_factory=list)
    web_pages: list[str] = field(default_factory=list)
    comments: list[dict[str, str]] = field(default_factory=list)
    diseases: list[dict[str, str]] = field(default_factory=list)
    species: list[dict[str, str]] = field(default_factory=list)
    hierarchy: list[dict[str, str]] = field(default_factory=list)
    same_individual: list[dict[str, str]] = field(default_factory=list)
    sex: str = ""
    age: str = ""
    category: str = ""
    created: str = ""
    last_updated: str = ""
    version: str = ""


CELL_LINES_HEADER = [
    "cellosaurus_id",
    "primary_name",
    "category",
    "sex",
    "age",
    "created",
    "last_updated",
    "version",
    "synonym_count",
    "secondary_accession_count",
    "cross_reference_count",
    "reference_count",
    "disease_count",
    "species_count",
    "hierarchy_count",
    "same_individual_count",
    "comment_count",
    "web_page_count",
]
SYNONYMS_HEADER = ["cellosaurus_id", "synonym"]
SECONDARY_ACCESSIONS_HEADER = ["cellosaurus_id", "secondary_accession"]
CROSS_REFERENCES_HEADER = ["cellosaurus_id", "resource", "identifier"]
REFERENCES_HEADER = ["cellosaurus_id", "source", "identifier"]
COMMENTS_HEADER = ["cellosaurus_id", "topic", "value"]
DISEASES_HEADER = ["cellosaurus_id", "resource", "identifier", "name"]
SPECIES_HEADER = ["cellosaurus_id", "taxonomy_id", "name"]
HIERARCHY_HEADER = ["cellosaurus_id", "parent_cellosaurus_id", "parent_name"]
SAME_INDIVIDUAL_HEADER = ["cellosaurus_id", "related_cellosaurus_id", "related_name"]
WEB_PAGES_HEADER = ["cellosaurus_id", "url"]


def parse_cellosaurus_file(path: Path) -> Iterator[CellosaurusEntry]:
    current = CellosaurusEntry()

    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")

            if not line:
                continue

            if line.startswith("ID   "):
                current = CellosaurusEntry(name=line[5:].strip())
                continue

            if current.name == "":
                continue

            if line == "//":
                yield current
                current = CellosaurusEntry()
                continue

            if len(line) < 5 or line[2:5] != "   ":
                continue

            code = line[:2]
            value = line[5:].strip()

            if code == "AC":
                current.accession = value
            elif code == "AS":
                current.secondary_accessions.extend(_split_semicolon_list(value))
            elif code == "SY":
                current.synonyms.extend(_split_semicolon_list(value))
            elif code == "DR":
                parsed = _parse_cross_reference(value)
                if parsed:
                    current.cross_references.append(parsed)
            elif code == "RX":
                current.references.extend(_parse_reference_identifiers(value))
            elif code == "WW":
                current.web_pages.append(value)
            elif code == "CC":
                current.comments.append(_parse_comment(value))
            elif code == "DI":
                parsed = _parse_disease(value)
                if parsed:
                    current.diseases.append(parsed)
            elif code == "OX":
                parsed = _parse_species(value)
                if parsed:
                    current.species.append(parsed)
            elif code == "HI":
                parsed = _parse_related_cell_line(value)
                if parsed:
                    current.hierarchy.append(parsed)
            elif code == "OI":
                parsed = _parse_related_cell_line(value)
                if parsed:
                    current.same_individual.append(parsed)
            elif code == "SX":
                current.sex = value
            elif code == "AG":
                current.age = value
            elif code == "CA":
                current.category = value
            elif code == "DT":
                parsed = _parse_dates(value)
                current.created = parsed.get("created", "")
                current.last_updated = parsed.get("last_updated", "")
                current.version = parsed.get("version", "")


def export_source_tables(input_path: Path, output_dir: Path) -> dict[str, object]:
     #如果父目录不存在，自动创建所有缺失的父目录；目录已存在不会报错

    output_dir.mkdir(parents=True, exist_ok=True)
    counters = Counter()
    categories = Counter()
    species_counter = Counter()
    diseases_counter = Counter()

    with _open_writer(output_dir / "cell_lines.csv", CELL_LINES_HEADER) as cell_lines_writer, \
        _open_writer(output_dir / "cell_line_synonyms.csv", SYNONYMS_HEADER) as synonyms_writer, \
        _open_writer(output_dir / "cell_line_secondary_accessions.csv", SECONDARY_ACCESSIONS_HEADER) as secondary_writer, \
        _open_writer(output_dir / "cell_line_cross_references.csv", CROSS_REFERENCES_HEADER) as cross_ref_writer, \
        _open_writer(output_dir / "cell_line_references.csv", REFERENCES_HEADER) as reference_writer, \
        _open_writer(output_dir / "cell_line_comments.csv", COMMENTS_HEADER) as comment_writer, \
        _open_writer(output_dir / "cell_line_diseases.csv", DISEASES_HEADER) as disease_writer, \
        _open_writer(output_dir / "cell_line_species.csv", SPECIES_HEADER) as species_writer, \
        _open_writer(output_dir / "cell_line_hierarchy.csv", HIERARCHY_HEADER) as hierarchy_writer, \
        _open_writer(output_dir / "cell_line_same_individual.csv", SAME_INDIVIDUAL_HEADER) as same_individual_writer, \
        _open_writer(output_dir / "cell_line_web_pages.csv", WEB_PAGES_HEADER) as web_page_writer:

        for entry in parse_cellosaurus_file(input_path):
            _write_source_rows(
                entry,
                cell_lines_writer,
                synonyms_writer,
                secondary_writer,
                cross_ref_writer,
                reference_writer,
                comment_writer,
                disease_writer,
                species_writer,
                hierarchy_writer,
                same_individual_writer,
                web_page_writer,
                counters,
                categories,
                species_counter,
                diseases_counter,
            )

    summary = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "counts": dict(counters),
        "top_categories": categories.most_common(10),
        "top_species": species_counter.most_common(10),
        "top_diseases": diseases_counter.most_common(10),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def _write_source_rows(
    entry: CellosaurusEntry,
    cell_lines_writer,
    synonyms_writer,
    secondary_writer,
    cross_ref_writer,
    reference_writer,
    comment_writer,
    disease_writer,
    species_writer,
    hierarchy_writer,
    same_individual_writer,
    web_page_writer,
    counters: Counter,
    categories: Counter,
    species_counter: Counter,
    disease_counter: Counter,
) -> None:
    counters["cell_lines"] += 1
    categories[entry.category] += 1

    cell_lines_writer.writerow(
        {
            "cellosaurus_id": entry.accession,
            "primary_name": entry.name,
            "category": entry.category,
            "sex": entry.sex,
            "age": entry.age,
            "created": entry.created,
            "last_updated": entry.last_updated,
            "version": entry.version,
            "synonym_count": len(entry.synonyms),
            "secondary_accession_count": len(entry.secondary_accessions),
            "cross_reference_count": len(entry.cross_references),
            "reference_count": len(entry.references),
            "disease_count": len(entry.diseases),
            "species_count": len(entry.species),
            "hierarchy_count": len(entry.hierarchy),
            "same_individual_count": len(entry.same_individual),
            "comment_count": len(entry.comments),
            "web_page_count": len(entry.web_pages),
        }
    )

    for synonym in entry.synonyms:
        counters["synonyms"] += 1
        synonyms_writer.writerow({"cellosaurus_id": entry.accession, "synonym": synonym})

    for secondary_accession in entry.secondary_accessions:
        counters["secondary_accessions"] += 1
        secondary_writer.writerow(
            {
                "cellosaurus_id": entry.accession,
                "secondary_accession": secondary_accession,
            }
        )

    for cross_reference in entry.cross_references:
        counters["cross_references"] += 1
        cross_ref_writer.writerow({"cellosaurus_id": entry.accession, **cross_reference})

    for reference in entry.references:
        counters["references"] += 1
        reference_writer.writerow({"cellosaurus_id": entry.accession, **reference})

    for comment in entry.comments:
        counters["comments"] += 1
        comment_writer.writerow({"cellosaurus_id": entry.accession, **comment})

    for disease in entry.diseases:
        counters["diseases"] += 1
        disease_counter[disease["name"]] += 1
        disease_writer.writerow({"cellosaurus_id": entry.accession, **disease})

    for item in entry.species:
        counters["species_links"] += 1
        species_counter[item["name"]] += 1
        species_writer.writerow({"cellosaurus_id": entry.accession, **item})

    for relation in entry.hierarchy:
        counters["hierarchy_links"] += 1
        hierarchy_writer.writerow(
            {
                "cellosaurus_id": entry.accession,
                "parent_cellosaurus_id": relation["related_accession"],
                "parent_name": relation["related_name"],
            }
        )

    for relation in entry.same_individual:
        counters["same_individual_links"] += 1
        same_individual_writer.writerow(
            {
                "cellosaurus_id": entry.accession,
                "related_cellosaurus_id": relation["related_accession"],
                "related_name": relation["related_name"],
            }
        )

    for url in entry.web_pages:
        counters["web_pages"] += 1
        web_page_writer.writerow({"cellosaurus_id": entry.accession, "url": url})


def _open_writer(path: Path, fieldnames: list[str]):
    handle = path.open("w", encoding="utf-8", newline="")  #w写入模式（会覆盖已有文件）；newline让python不要处理换行，csv会自己处理
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    return _CsvWriterContext(handle, writer)


class _CsvWriterContext:
    def __init__(self, handle, writer: csv.DictWriter):
        self.handle = handle
        self.writer = writer

    def __enter__(self) -> csv.DictWriter:
        return self.writer

    def __exit__(self, exc_type, exc, tb) -> None:
        self.handle.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse Cellosaurus flat file into source-level CSV tables."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/cellosaurus/cellosaurus.txt"),
        help="Path to the Cellosaurus flat file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/intermediate/normalized/cellosaurus"),
        help="Directory where source-level parsed tables will be written.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = export_source_tables(args.input, args.output_dir)
    print(
        json.dumps(
            {
                "cell_lines": summary["counts"].get("cell_lines", 0),
                "cross_references": summary["counts"].get("cross_references", 0),
                "diseases": summary["counts"].get("diseases", 0),
                "output_dir": summary["output_dir"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _split_semicolon_list(value: str) -> list[str]:
    items = [item.strip() for item in value.split(";")]
    return [item for item in items if item]


def _parse_cross_reference(value: str) -> dict[str, str] | None:
    parts = [part.strip() for part in value.split(";") if part.strip()]
    if len(parts) < 2:
        return None
    return {"resource": parts[0], "identifier": "; ".join(parts[1:])}


def _parse_reference_identifiers(value: str) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for part in value.split(";"):
        item = part.strip()
        if not item or "=" not in item:
            continue
        source, identifier = item.split("=", 1)
        results.append({"source": source.strip(), "identifier": identifier.strip()})
    return results


def _parse_comment(value: str) -> dict[str, str]:
    if ":" not in value:
        return {"topic": "", "value": value.strip()}
    topic, detail = value.split(":", 1)
    return {"topic": topic.strip(), "value": detail.strip()}


def _parse_disease(value: str) -> dict[str, str] | None:
    parts = [part.strip() for part in value.split(";") if part.strip()]
    if len(parts) < 3:
        return None
    return {
        "resource": parts[0],
        "identifier": parts[1],
        "name": "; ".join(parts[2:]),
    }


def _parse_species(value: str) -> dict[str, str] | None:
    if ";" not in value:
        return None
    left, right = value.split("!", 1) if "!" in value else (value, "")
    left_parts = [part.strip() for part in left.split(";") if part.strip()]
    if not left_parts:
        return None

    taxon = left_parts[0]
    taxonomy_id = taxon.split("=", 1)[1].strip() if "=" in taxon else taxon
    return {
        "taxonomy_id": taxonomy_id,
        "name": right.strip(),
    }


def _parse_related_cell_line(value: str) -> dict[str, str] | None:
    if "!" not in value:
        return None
    accession, name = value.split("!", 1)
    return {
        "related_accession": accession.strip(),
        "related_name": name.strip(),
    }


def _parse_dates(value: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for part in value.split(";"):
        item = part.strip()
        if not item or ":" not in item:
            continue
        key, item_value = item.split(":", 1)
        normalized_key = key.strip().lower().replace(" ", "_")
        parsed[normalized_key] = item_value.strip()
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
