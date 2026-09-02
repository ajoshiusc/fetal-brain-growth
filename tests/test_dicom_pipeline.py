from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import BasicTextSRStorage, ExplicitVRLittleEndian, generate_uid
import pytest

from fetal_brain_growth.dicom_pipeline import (
    _absolute_executable,
    build_svr_command,
    extract_gestational_age,
    parse_gestational_age_override,
)


def _write_sr(path: Path, text: str, *, study_uid: str, study_date: str = "20260702") -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = BasicTextSRStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = BasicTextSRStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = study_uid
    dataset.SeriesInstanceUID = generate_uid()
    dataset.Modality = "SR"
    dataset.StudyDate = study_date
    dataset.ContentDate = study_date
    dataset.ContentTime = datetime.now().strftime("%H%M%S")
    content = Dataset()
    content.ValueType = "TEXT"
    content.TextValue = text
    dataset.ContentSequence = Sequence([content])
    dataset.save_as(path, enforce_file_format=True)


def test_extracts_precise_ga_from_structured_report_and_flags_shifted_dates(tmp_path: Path):
    _write_sr(
        tmp_path / "report",
        (
            "The estimated gestational age is 33 weeks and 1 day by estimated due date "
            "of 10/18/2026. Gestational age: 33 weeks."
        ),
        study_uid=generate_uid(),
    )
    result = extract_gestational_age(tmp_path)
    assert result["weeks"] == 33
    assert result["days"] == 1
    assert result["decimal_weeks"] == pytest.approx(33 + 1 / 7)
    assert result["source_file"] == "report"
    assert result["source_tag"] == "(0040,A160) Text Value"
    assert result["warnings"]


def test_rejects_multiple_dicom_studies(tmp_path: Path):
    _write_sr(tmp_path / "one.dcm", "Gestational age: 30 weeks", study_uid=generate_uid())
    _write_sr(tmp_path / "two.dcm", "Gestational age: 30 weeks", study_uid=generate_uid())
    with pytest.raises(ValueError, match="Found 2 DICOM studies"):
        extract_gestational_age(tmp_path)


def test_ga_override_and_svr_command_are_deterministic(tmp_path: Path):
    assert parse_gestational_age_override("33+1")["total_days"] == 232
    command = build_svr_command(
        svr_python=Path("/opt/svr/.venv/bin/python"),
        svr_root=Path("/opt/svr"),
        dicom_dir=tmp_path / "dicoms",
        output_parent=tmp_path / "outputs",
        study_name="case_SVR",
        device=0,
        max_series=4,
        batch_size_seg=8,
        dilation_radius_seg=1.5,
        series_keywords=("brain", "tse"),
        echo_time=90.0,
    )
    assert command[:2] == ["/opt/svr/.venv/bin/python", "/opt/svr/run_svr_gpu.py"]
    assert command.count("--include-series-keyword") == 2
    assert command[-2:] == ["--te", "90.0"]


def test_virtualenv_executable_symlink_is_not_dereferenced(tmp_path: Path):
    environment = tmp_path / ".venv" / "bin"
    environment.mkdir(parents=True)
    executable = environment / "python"
    executable.symlink_to("/usr/bin/python3")

    assert _absolute_executable(executable) == executable
    assert _absolute_executable(executable) != executable.resolve()
