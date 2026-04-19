import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend import data_access  # noqa: E402


def test_control_room_payload_includes_dashboard_contract(monkeypatch):
    history = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-04-13 00:00:00",
                    "2026-04-13 00:01:00",
                    "2026-04-13 00:02:00",
                ],
                utc=True,
            ),
            "machine_id_normalized": ["M231", "M231", "M231"],
            "is_scrap_actual": [0, 1, 0],
            "Cycle_time": [18.0, 19.5, 20.0],
            "Injection_pressure": [1150.0, 1180.0, 1200.0],
        }
    )

    monkeypatch.setattr(
        data_access,
        "_build_machine_feb_history",
        lambda machine_norm: (
            history.copy(),
            {
                "id": machine_norm,
                "display_id": "M-231",
                "machine_number": "231",
                "tool_id": "T-041",
                "part_number": "P-001",
                "name": "Injection Molder M-231",
                "machine_type": "High Precision",
                "machine_definition": "M231-TEST",
            },
        ),
    )
    monkeypatch.setattr(
        data_access,
        "calculate_dynamic_limits",
        lambda df: {
            "Cycle_time": {"min": 10.0, "max": 30.0},
            "Injection_pressure": {"min": 1000.0, "max": 1300.0},
        },
    )
    monkeypatch.setattr(
        data_access,
        "build_realtime_model_vector",
        lambda df, machine_norm="", strict=False: {
            "Cycle_time": 20.0,
            "Injection_pressure": 1200.0,
        },
    )
    monkeypatch.setattr(data_access, "unified_predict_scrap", lambda machine_norm, sensor_input: 0.84)
    monkeypatch.setattr(
        data_access,
        "unified_get_model_and_features",
        lambda machine_norm: (
            object(),
            ("Cycle_time", "Injection_pressure"),
        ),
    )
    monkeypatch.setattr(
        data_access,
        "compute_root_causes",
        lambda model, feature_row, feature_names: [
            {
                "cause": "Cycle Time Instability",
                "impact": 0.47,
                "category": "Cycle Time Instability",
                "total_impact": 0.47,
                "risk_increasing": 0.47,
                "risk_decreasing": 0.0,
                "top_parameters": [{"parameter": "Cycle_time", "impact": 0.47}],
            }
        ],
    )
    monkeypatch.setattr(
        data_access,
        "_generate_future_horizon",
        lambda machine_df, n_steps=30: [
            {
                "timestamp": 1712966580000,
                "risk_score": 0.9,
                "is_future": True,
                "type": "future",
                "horizon_minutes": 5,
                "is_scrap_actual": 0,
                "sensors": {},
            }
        ],
    )

    payload = data_access.build_control_room_payload("M231", time_window=60, future_window=30)

    assert set(["machine_info", "summary_stats", "current_health", "root_causes", "telemetry_grid", "timeline", "safe_limits"]).issubset(payload.keys())
    assert payload["current_health"]["status"] in {"HIGH", "CRITICAL"}
    assert payload["current_health"]["root_causes"] == ["Cycle Time Instability"]
    assert payload["summary_stats"]["past_scrap_detected"] == 1
    assert payload["summary_stats"]["future_scrap_predicted"] == 1
    assert len(payload["root_causes"]) == 1
    assert payload["root_causes"][0]["top_parameters"][0]["parameter"] == "Cycle_time"

    telemetry = payload["telemetry_grid"]
    assert telemetry
    assert any(row["sensor"] == "Cycle_time" and row["is_root_cause"] for row in telemetry)
    assert any(row["sensor"] == "Injection_pressure" for row in telemetry)

    timeline = payload["timeline"]
    assert timeline
    assert any(point["type"] == "past" for point in timeline)
    assert any(point["type"] == "future" for point in timeline)
