"""Unit Verification for Brain-Isomorphic Cognitive Architecture (BICA) Subsystems."""
from __future__ import annotations

from pathlib import Path

from genesis_bica.fast_episodic_memory import FastEpisodicMemory
from genesis_bica.homeostatic_interoceptor import HomeostaticInteroceptor


def test_fast_episodic_memory_one_shot_belief_revision(tmp_path: Path) -> None:
    """CA3 Fast Episodic Memory supports 1-shot insertion & ATMS contradiction revision."""
    db_path = tmp_path / "episodic_test.sqlite"
    mem = FastEpisodicMemory(db_path=db_path, max_slots=64)

    mem.write("Project lead is Dr. Nguyen.", entity="project_lead", attribute="name", surprise=0.95)
    hits_1 = mem.retrieve("Who is the project lead?", entity="project_lead", top_k=2)
    assert hits_1 and "Nguyen" in hits_1[0].text

    # 1-shot belief revision supersedes stale belief cleanly
    mem.write("Project lead is now Dr. Tran.", entity="project_lead", attribute="name", surprise=0.99)
    hits_2 = mem.retrieve("Who is the project lead?", entity="project_lead", top_k=2)
    assert hits_2 and "Tran" in hits_2[0].text
    mem.close()


def test_homeostatic_interoceptor_telemetry_bounds() -> None:
    """Interoceptor maps host telemetry to bounded neuromodulators DA/NE/ACh/5-HT."""
    ic = HomeostaticInteroceptor(
        drive_path=".",
        telemetry_override={
            "ac_plugged": True,
            "battery_pct": 90.0,
            "cpu_load_pct": 25.0,
            "mem_free_gb": 8.0,
            "disk_e_free_gb": 50.0,
            "disk_e_usage_pct": 40.0,
        },
    )
    st = ic.sample()
    for name in ("da", "ne", "ach", "serotonin"):
        val = getattr(st, name)
        assert 0.0 <= val <= 1.0, f"{name} out of bounds: {val}"
    st2 = ic.update_feedback(cert_ok=True, contradiction=False, surprise=0.2)
    assert 0.0 <= st2.da <= 1.0 and 0.0 <= st2.ht5 <= 1.0
