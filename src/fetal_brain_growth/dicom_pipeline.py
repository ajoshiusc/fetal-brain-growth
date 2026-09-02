"""End-to-end DICOM to SVR, FetalSynthSeg, and growth-report pipeline."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Sequence

import pandas as pd

from .case_report import save_case_report
from .labels import FETA_MATCHED_REFERENCE_REGIONS
from .references import score_against_curves
from .volumetry import measure_segmentation


DEFAULT_SVR_ROOT = Path(os.environ.get("SVR_GPU_ROOT", "/home/ajoshi/Projects/svr_gpu"))
DEFAULT_REFERENCE_RELATIVE = Path(
    "meeting_outputs/feta_10_cases_matched/feta_neurotypical_reference_curves.csv"
)
DEFAULT_CHECKPOINT_RELATIVE = Path("models/KISPI-all_fss.ckpt")

_DIRECT_GA_PATTERNS = (
    re.compile(
        r"\b(?:estimated\s+)?gestational\s+age(?:\s+(?:is|of))?\s*[:=\-]?\s*"
        r"(?P<weeks>\d{1,2})\s*(?:weeks?|wks?|wk|w)"
        r"(?:\s*(?:and|\+)\s*(?P<days>[0-6])\s*(?:days?|d)?)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:GA|EGA)\s*[:=\-]\s*(?P<weeks>\d{1,2})"
        r"(?:\s*(?:weeks?|wks?|wk|w))?\s*(?:\+\s*(?P<days>[0-6]))?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?P<weeks>\d{1,2})\s*(?:weeks?|wks?|wk|w)"
        r"(?:\s*(?:and|\+)\s*(?P<days>[0-6])\s*(?:days?|d)?)?"
        r"\s+gestational\s+age\b",
        re.IGNORECASE,
    ),
)
_DUE_DATE_PATTERN = re.compile(
    r"\b(?:estimated\s+)?due\s+date(?:\s+(?:of|is))?\s*[:=\-]?\s*"
    r"(?P<date>\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2}|\d{8})\b",
    re.IGNORECASE,
)
_TEXT_KEYWORDS = (
    "PatientComments",
    "AdditionalPatientHistory",
    "StudyDescription",
    "RequestedProcedureDescription",
    "PerformedProcedureStepDescription",
    "ReasonForRequestedProcedure",
    "AdmittingDiagnosesDescription",
)


@dataclass(frozen=True)
class GACandidate:
    """One gestational-age value recovered from DICOM metadata or report text."""

    total_days: int
    weeks: int
    days: int
    method: str
    score: int
    source_file: str
    source_tag: str
    evidence: str


def _collapse_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def _date_from_dicom(value: str) -> date | None:
    value = value.strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def _candidate(
    total_days: int,
    *,
    method: str,
    score: int,
    source_file: Path,
    source_tag: str,
    evidence: str,
) -> GACandidate | None:
    if not 14 * 7 <= total_days <= 45 * 7:
        return None
    return GACandidate(
        total_days=total_days,
        weeks=total_days // 7,
        days=total_days % 7,
        method=method,
        score=score,
        source_file=str(source_file),
        source_tag=source_tag,
        evidence=_collapse_text(evidence)[:300],
    )


def _direct_candidates(text: str, source_file: Path, source_tag: str) -> list[GACandidate]:
    collapsed = _collapse_text(text)
    candidates: list[GACandidate] = []
    seen_spans: set[tuple[int, int]] = set()
    for pattern in _DIRECT_GA_PATTERNS:
        for match in pattern.finditer(collapsed):
            if match.span() in seen_spans:
                continue
            seen_spans.add(match.span())
            weeks = int(match.group("weeks"))
            day_text = match.groupdict().get("days")
            days = int(day_text) if day_text is not None else 0
            value = _candidate(
                weeks * 7 + days,
                method="explicit_gestational_age",
                score=100 if day_text is not None else 90,
                source_file=source_file,
                source_tag=source_tag,
                evidence=match.group(0),
            )
            if value is not None:
                candidates.append(value)
    return candidates


def _due_date_candidates(
    text: str,
    study_date: date | None,
    source_file: Path,
    source_tag: str,
) -> list[GACandidate]:
    if study_date is None:
        return []
    candidates: list[GACandidate] = []
    for match in _DUE_DATE_PATTERN.finditer(_collapse_text(text)):
        due_date = _date_from_dicom(match.group("date"))
        if due_date is None:
            continue
        total_days = 280 - (due_date - study_date).days
        value = _candidate(
            total_days,
            method="estimated_due_date_and_study_date",
            score=50,
            source_file=source_file,
            source_tag=source_tag,
            evidence=f"{match.group(0)}; StudyDate={study_date:%Y%m%d}",
        )
        if value is not None:
            candidates.append(value)
    return candidates


def parse_gestational_age_override(value: str) -> dict[str, object]:
    """Parse ``33+1``, ``33w1d``, or decimal-week GA supplied by a user."""

    cleaned = value.strip().lower()
    match = re.fullmatch(r"(?P<w>\d{1,2})\s*(?:w(?:eeks?)?)?\s*\+\s*(?P<d>[0-6])(?:\s*d)?", cleaned)
    if match is None:
        match = re.fullmatch(r"(?P<w>\d{1,2})\s*w(?:eeks?)?\s*(?P<d>[0-6])\s*d", cleaned)
    if match is not None:
        total_days = int(match.group("w")) * 7 + int(match.group("d"))
    else:
        try:
            total_days = int(round(float(cleaned) * 7))
        except ValueError as error:
            raise ValueError("Gestational age must look like 33+1, 33w1d, or 33.14.") from error
    if not 14 * 7 <= total_days <= 45 * 7:
        raise ValueError("Gestational age must be between 14 and 45 weeks.")
    return {
        "weeks": total_days // 7,
        "days": total_days % 7,
        "total_days": total_days,
        "decimal_weeks": total_days / 7.0,
        "method": "user_override",
        "source_file": None,
        "source_tag": None,
        "evidence": value,
        "warnings": [],
        "candidates": [],
    }


def extract_gestational_age(dicom_dir: str | Path) -> dict[str, object]:
    """Extract the most specific GA from one DICOM study with auditable evidence."""

    try:
        import pydicom
    except ImportError as error:  # pragma: no cover - depends on optional runtime
        raise ImportError("Install pydicom or install this project with its pipeline dependencies.") from error

    dicom_dir = Path(dicom_dir).expanduser().resolve()
    if not dicom_dir.is_dir():
        raise FileNotFoundError(f"DICOM directory does not exist: {dicom_dir}")
    candidates: list[GACandidate] = []
    study_uids: set[str] = set()
    study_dates: Counter[str] = Counter()
    valid_files = 0
    unreadable_files = 0
    last_menstrual_dates: list[tuple[date, Path]] = []
    deferred_text: dict[tuple[str, str], Path] = {}

    for path in sorted(item for item in dicom_dir.rglob("*") if item.is_file()):
        try:
            dataset = pydicom.dcmread(str(path), stop_before_pixels=True)
        except Exception:
            unreadable_files += 1
            continue
        if not getattr(dataset, "SOPClassUID", None):
            continue
        valid_files += 1
        study_uid = str(getattr(dataset, "StudyInstanceUID", "")).strip()
        if study_uid:
            study_uids.add(study_uid)
        raw_study_date = str(getattr(dataset, "StudyDate", "")).strip()
        if raw_study_date:
            study_dates[raw_study_date] += 1
        lmp = _date_from_dicom(str(getattr(dataset, "LastMenstrualDate", "")))
        if lmp is not None:
            last_menstrual_dates.append((lmp, path))

        for keyword in _TEXT_KEYWORDS:
            if hasattr(dataset, keyword):
                text = str(getattr(dataset, keyword)).strip()
                if text:
                    deferred_text.setdefault((text, keyword), path)
        modality = str(getattr(dataset, "Modality", ""))
        elements = dataset.iterall() if modality == "SR" else iter(dataset)
        for element in elements:
            name = str(element.name)
            if element.tag == (0x0040, 0xA160):
                deferred_text.setdefault((str(element.value), "(0040,A160) Text Value"), path)
            elif "gestational age" in name.lower():
                deferred_text.setdefault((str(element.value), f"{element.tag} {name}"), path)

    if valid_files == 0:
        raise ValueError(f"No readable DICOM objects were found under {dicom_dir}.")
    if len(study_uids) > 1:
        raise ValueError(
            f"Found {len(study_uids)} DICOM studies. Place one study in the input directory per run."
        )
    study_date_value = study_dates.most_common(1)[0][0] if study_dates else ""
    study_date = _date_from_dicom(study_date_value)
    for (text, source_tag), source_file in deferred_text.items():
        candidates.extend(_direct_candidates(text, source_file, source_tag))
        candidates.extend(_due_date_candidates(text, study_date, source_file, source_tag))
    if study_date is not None:
        for lmp, source_file in last_menstrual_dates:
            value = _candidate(
                (study_date - lmp).days,
                method="last_menstrual_date_and_study_date",
                score=60,
                source_file=source_file,
                source_tag="LastMenstrualDate + StudyDate",
                evidence=f"LastMenstrualDate={lmp:%Y%m%d}; StudyDate={study_date:%Y%m%d}",
            )
            if value is not None:
                candidates.append(value)

    unique_candidates = list(
        {
            (
                item.total_days,
                item.method,
                item.source_file,
                item.source_tag,
                item.evidence,
            ): item
            for item in candidates
        }.values()
    )
    if not unique_candidates:
        raise ValueError(
            "No gestational age was found in DICOM text, LMP, or due-date metadata. "
            "Pass --gestational-age (for example, 33+1)."
        )
    highest_score = max(item.score for item in unique_candidates)
    best = [item for item in unique_candidates if item.score == highest_score]
    most_common_days = Counter(item.total_days for item in best).most_common(1)[0][0]
    selected = sorted(
        (item for item in best if item.total_days == most_common_days),
        key=lambda item: (item.source_file, item.source_tag, item.evidence),
    )[0]

    warnings: list[str] = []
    explicit_days = [
        item.total_days for item in unique_candidates if item.method == "explicit_gestational_age"
    ]
    if explicit_days and max(explicit_days) - min(explicit_days) > 7:
        warnings.append("Explicit gestational-age statements differ by more than seven days.")
    date_derived = [
        item for item in unique_candidates if item.method != "explicit_gestational_age"
    ]
    if any(abs(item.total_days - selected.total_days) > 7 for item in date_derived):
        warnings.append(
            "Date-derived GA disagrees with the explicit report age by more than seven days; "
            "DICOM dates may have been shifted or deidentified. The explicit report age was used."
        )

    relative_source = Path(selected.source_file)
    try:
        relative_source = relative_source.relative_to(dicom_dir)
    except ValueError:
        pass
    study_uid = next(iter(study_uids), "")
    return {
        "weeks": selected.weeks,
        "days": selected.days,
        "total_days": selected.total_days,
        "decimal_weeks": selected.total_days / 7.0,
        "method": selected.method,
        "source_file": str(relative_source),
        "source_tag": selected.source_tag,
        "evidence": selected.evidence,
        "warnings": warnings,
        "candidates": [asdict(item) for item in sorted(unique_candidates, key=lambda item: (-item.score, item.total_days))],
        "dicom_file_count": valid_files,
        "unreadable_file_count": unreadable_files,
        "study_count": len(study_uids),
        "study_uid_sha256": hashlib.sha256(study_uid.encode()).hexdigest() if study_uid else None,
        "study_date": study_date_value or None,
    }


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return result or "fetal_case"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_executable(path: str | Path) -> Path:
    """Make an executable path absolute without dereferencing virtualenv symlinks.

    Python uses the location from which its executable was invoked to discover a
    virtual environment.  ``Path.resolve()`` changes ``.venv/bin/python`` into
    the system interpreter and therefore silently bypasses the virtualenv.
    """

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.absolute()


def build_svr_command(
    *,
    svr_python: Path,
    svr_root: Path,
    dicom_dir: Path,
    output_parent: Path,
    study_name: str,
    device: int,
    max_series: int,
    batch_size_seg: int | None,
    dilation_radius_seg: float | None,
    series_keywords: Sequence[str],
    echo_time: float | None,
) -> list[str]:
    """Build the external, pinned SVR command without executing it."""

    command = [
        str(svr_python),
        str(svr_root / "run_svr_gpu.py"),
        str(dicom_dir),
        str(output_parent),
        "--study-name",
        study_name,
        "--device",
        str(device),
        "--max-series",
        str(max_series),
        "--keep-temp",
    ]
    if batch_size_seg is not None:
        command.extend(("--batch-size-seg", str(batch_size_seg)))
    if dilation_radius_seg is not None:
        command.extend(("--dilation-radius-seg", str(dilation_radius_seg)))
    for keyword in series_keywords:
        command.extend(("--include-series-keyword", keyword))
    if echo_time is not None:
        command.extend(("--te", str(echo_time)))
    return command


def _run_logged(command: Sequence[str], *, cwd: Path, log_path: Path, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(command) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}; see {log_path}.")


def _find_new_svr_run(output_parent: Path, study_name: str, before: set[Path]) -> Path:
    candidates = [
        path for path in output_parent.glob(f"{study_name}_*") if path.is_dir() and path not in before
    ]
    if not candidates:
        raise RuntimeError(f"SVR completed but no new {study_name}_* run directory was found.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _resolve_reference(project_root: Path, value: str | None) -> Path:
    candidate = value or os.environ.get("FBG_REFERENCE_CURVES")
    path = Path(candidate).expanduser() if candidate else project_root / DEFAULT_REFERENCE_RELATIVE
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"FeTA-matched reference curves were not found at {path}. "
            "Pass --reference-curves or set FBG_REFERENCE_CURVES."
        )
    return path


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    project_root = Path(__file__).resolve().parents[2]
    dicom_dir = Path(args.dicom_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    subject_id = args.subject_id or dicom_dir.parent.name or "fetal_case"
    slug = _slug(subject_id)

    print("Step 1/5: extracting gestational age and validating one DICOM study", flush=True)
    ga = (
        parse_gestational_age_override(args.gestational_age)
        if args.gestational_age
        else extract_gestational_age(dicom_dir)
    )
    print(
        f"Gestational age: {ga['weeks']} weeks {ga['days']} "
        f"{'day' if ga['days'] == 1 else 'days'} "
        f"({ga['decimal_weeks']:.6f} weeks) from {ga['method']}",
        flush=True,
    )
    if ga.get("source_file"):
        print(f"GA source: {ga['source_file']} — {ga['source_tag']}", flush=True)
    for warning in ga.get("warnings", []):
        print(f"WARNING: {warning}", flush=True)

    svr_root = Path(args.svr_root).expanduser().resolve()
    svr_python = _absolute_executable(
        args.svr_python if args.svr_python else svr_root / ".venv/bin/python"
    )
    if not (svr_root / "run_svr_gpu.py").is_file():
        raise FileNotFoundError(f"SVR runner was not found under {svr_root}.")
    if not svr_python.is_file():
        raise FileNotFoundError(f"SVR Python was not found: {svr_python}")

    svr_parent = output_dir / "svr_runs"
    svr_parent.mkdir(parents=True, exist_ok=True)
    study_name = f"{slug}_SVR"
    svr_command = build_svr_command(
        svr_python=svr_python,
        svr_root=svr_root,
        dicom_dir=dicom_dir,
        output_parent=svr_parent,
        study_name=study_name,
        device=args.svr_device,
        max_series=args.max_series,
        batch_size_seg=args.batch_size_seg,
        dilation_radius_seg=args.dilation_radius_seg,
        series_keywords=args.series_keyword,
        echo_time=args.echo_time,
    )
    if args.svr_nifti:
        print("Step 2/5: using the explicitly supplied existing SVR NIfTI", flush=True)
        svr_nifti = Path(args.svr_nifti).expanduser().resolve()
        svr_run_dir: Path | None = None
        if not svr_nifti.is_file():
            raise FileNotFoundError(f"SVR NIfTI does not exist: {svr_nifti}")
    elif args.dry_run:
        print("Step 2/5: dry run; SVR command would be:")
        print(" ".join(svr_command))
        return {"gestational_age": ga, "svr_command": svr_command, "dry_run": True}
    else:
        print("Step 2/5: reconstructing the fetal brain from selected DICOM series", flush=True)
        before = set(svr_parent.glob(f"{study_name}_*"))
        _run_logged(svr_command, cwd=svr_root, log_path=output_dir / "svr.log")
        svr_run_dir = _find_new_svr_run(svr_parent, study_name, before)
        svr_nifti = svr_run_dir / "out/tmp/svr_output.nii.gz"
        if not svr_nifti.is_file():
            raise FileNotFoundError(f"SVR did not produce the expected NIfTI: {svr_nifti}")

    checkpoint = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else (project_root / DEFAULT_CHECKPOINT_RELATIVE).resolve()
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"FetalSynthSeg checkpoint does not exist: {checkpoint}")
    fss_python = _absolute_executable(args.fss_python if args.fss_python else sys.executable)
    segmentation = output_dir / f"{slug}_fetalsynthseg.nii.gz"
    segmentation_metadata = output_dir / f"{slug}_fetalsynthseg.json"
    fss_command = [
        str(fss_python),
        "-m",
        "fetal_brain_growth.segmentation",
        "--input",
        str(svr_nifti),
        "--output",
        str(segmentation),
        "--checkpoint",
        str(checkpoint),
        "--device",
        args.fss_device,
        "--metadata",
        str(segmentation_metadata),
    ]
    print("Step 3/5: running checksum-verified FetalSynthSeg", flush=True)
    fss_env = os.environ.copy()
    existing_pythonpath = fss_env.get("PYTHONPATH")
    fss_env["PYTHONPATH"] = str(project_root / "src") + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    _run_logged(fss_command, cwd=project_root, log_path=output_dir / "fetalsynthseg.log", env=fss_env)

    print("Step 4/5: measuring tissues and scoring FeTA-matched growth curves", flush=True)
    curves_path = _resolve_reference(project_root, args.reference_curves)
    curves = pd.read_csv(curves_path)
    age = float(ga["decimal_weeks"])
    low_age = float(curves.gestational_age_weeks.min())
    high_age = float(curves.gestational_age_weeks.max())
    if not low_age <= age <= high_age:
        raise ValueError(
            f"GA {age:.3f} weeks is outside the reference range {low_age:.3f}-{high_age:.3f}; "
            "growth charts will not be extrapolated."
        )
    tissues, aggregates, qc = measure_segmentation(
        segmentation,
        subject_id=subject_id,
        gestational_age_weeks=age,
    )
    case_volumes = pd.concat(
        [
            aggregates.loc[aggregates.region.isin(("total_brain", "intracranial_volume"))],
            tissues,
        ],
        ignore_index=True,
        sort=False,
    )
    scores = score_against_curves(case_volumes, curves, definition_guard=False)
    tissue_path = output_dir / f"{slug}_tissue_volumes.csv"
    aggregate_path = output_dir / f"{slug}_aggregate_volumes.csv"
    score_path = output_dir / f"{slug}_growth_scores.csv"
    qc_path = output_dir / f"{slug}_segmentation_qc.json"
    tissues.to_csv(tissue_path, index=False)
    aggregates.to_csv(aggregate_path, index=False)
    scores.to_csv(score_path, index=False)
    qc_path.write_text(json.dumps(qc, indent=2) + "\n")

    print("Step 5/5: rendering the high-resolution radiology growth report", flush=True)
    report = output_dir / f"{slug}_radiology_report.png"
    save_case_report(
        svr_nifti,
        segmentation,
        curves,
        scores,
        report,
        subject_id=subject_id,
        gestational_age_weeks=age,
        segmentation_source="FetalSynthSeg automatic prediction",
        regions=FETA_MATCHED_REFERENCE_REGIONS,
        dpi=args.report_dpi,
    )

    provenance = {
        "subject_id": subject_id,
        "dicom_directory": str(dicom_dir),
        "gestational_age": ga,
        "svr": {
            "root": str(svr_root),
            "command": svr_command if not args.svr_nifti else None,
            "run_directory": str(svr_run_dir) if svr_run_dir else None,
            "nifti": str(svr_nifti),
            "nifti_sha256": _sha256(svr_nifti),
        },
        "fetalsynthseg": {
            "command": fss_command,
            "segmentation": str(segmentation),
            "segmentation_sha256": _sha256(segmentation),
            "metadata": str(segmentation_metadata),
        },
        "reference_curves": str(curves_path),
        "reference_curves_sha256": _sha256(curves_path),
        "segmentation_qc": qc,
        "outputs": {
            "report": str(report),
            "tissues": str(tissue_path),
            "aggregates": str(aggregate_path),
            "scores": str(score_path),
            "qc": str(qc_path),
        },
        "intended_use": "Research use only; visual QC and expert review required.",
    }
    provenance_path = output_dir / f"{slug}_pipeline_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Complete: {report}", flush=True)
    return {**provenance["outputs"], "provenance": str(provenance_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruct fetal MRI from DICOM, run FetalSynthSeg, and create a growth report."
    )
    parser.add_argument("dicom_dir", help="Directory containing one DICOM study, in any layout")
    parser.add_argument("output_dir", help="Directory for the SVR run, segmentation, tables, and PNG")
    parser.add_argument("--subject-id", help="Label shown on the report; defaults to the DICOM parent directory")
    parser.add_argument(
        "--gestational-age",
        help="Override missing DICOM GA, such as 33+1, 33w1d, or 33.14",
    )
    parser.add_argument("--svr-root", default=str(DEFAULT_SVR_ROOT), help="Directory containing run_svr_gpu.py")
    parser.add_argument("--svr-python", help="Python executable for the SVR project")
    parser.add_argument("--svr-device", type=int, default=0, help="SVR CUDA device; use -1 for CPU")
    parser.add_argument("--max-series", type=int, default=4, help="Maximum selected T2 brain series")
    parser.add_argument("--batch-size-seg", type=int, default=8, help="SVR brain-mask inference batch size")
    parser.add_argument("--dilation-radius-seg", type=float, help="SVR brain-mask dilation in millimeters")
    parser.add_argument(
        "--series-keyword",
        action="append",
        default=[],
        help="Require this SeriesDescription keyword; repeat to require all keywords",
    )
    parser.add_argument("--echo-time", type=float, help="Select DICOM series within 1 ms of this TE")
    parser.add_argument("--svr-nifti", help="Reuse an existing SVR NIfTI instead of reconstructing")
    parser.add_argument("--fss-python", help="Python executable with torch and MONAI; defaults to this interpreter")
    parser.add_argument("--fss-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint", help="FetalSynthSeg checkpoint path")
    parser.add_argument("--reference-curves", help="FeTA-matched reference-curves CSV")
    parser.add_argument("--report-dpi", type=int, default=450)
    parser.add_argument("--dry-run", action="store_true", help="Validate DICOM GA and print the SVR command only")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        result = run_pipeline(args)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
