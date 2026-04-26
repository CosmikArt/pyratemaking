"""Smoke tests — verify the package is importable and version is set."""


def test_import():
    import pyratemaking
    assert pyratemaking.__version__ == "0.0.1"


def test_classes_importable():
    from pyratemaking import (
        FrequencySeverityModel,
        BaseRateCalculator,
        ClassificationRatemaker,
        ExposureHandler,
        LiftChart,
        AEDiagnostic,
    )
    # Instantiation should not raise
    FrequencySeverityModel()
    BaseRateCalculator()
    ClassificationRatemaker()
    LiftChart()
    AEDiagnostic()
