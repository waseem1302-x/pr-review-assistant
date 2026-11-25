"""Adaptive rules must stay quiet on files outside the pull request."""

import json
import os
import shutil
import subprocess

import pytest
import yaml

import src.agent.rules as rules_module
from src.agent.budget import Budget
from src.agent.context import ReviewContext
from src.agent.findings import FindingCollector
from src.agent.loop import run_agent
from src.agent.rules import RuleCollector
from src.agent.toolbelt import Toolbelt
from src.diff.patch import DiffMap
from src.llm.base import LLMConfig, ToolCall
from tests.fakes import FakeProvider, call, turn


RULE = """
id: decorator-order
languages: [python]
severity: WARNING
message: Check decorator order
pattern: |
  @classmethod
  @cache
  def $F(...): ...
"""


def test_noisy_rule_is_rejected_and_feedback_reaches_author(tmp_path, monkeypatch):
    results = [
        {"check_id": "tmp.config.decorator-order", "path": "accepted.py", "start": {"line": 2}},
        {"check_id": "tmp.config.decorator-order", "path": "accepted.py", "start": {"line": 8}},
    ]

    def run(cmd, **kwargs):
        if cmd[0] == "git":
            return subprocess.CompletedProcess(cmd, 0, "accepted.py\0changed.py\0", "")
        if "--validate" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"results": results}), "")

    monkeypatch.setattr(subprocess, "run", run)
    context = ReviewContext(
        workspace=str(tmp_path),
        diff=DiffMap.from_pull_files({"changed.py": {"patch": "", "status": "modified"}}),
    )
    findings, rules, budget = FindingCollector(), RuleCollector(), Budget()
    belt = Toolbelt(context, findings, budget, rule_collector=rules)
    llm = FakeProvider([
        turn(call("write_rule", rule_yaml=RULE, rationale="convention")),
        turn(call("finish", summary="finished")),
    ])
    run_agent(llm, LLMConfig(), "author", "write a rule", belt, findings, budget)

    rule = rules.rules[0]
    expected = (
        "fires on 2 line(s) of code this project already accepts (e.g. accepted.py:2); "
        "a rule must stay quiet on accepted code"
    )
    assert not rule.valid
    assert rule.baseline_hits == 2
    assert rule.rejected_because == expected
    assert rule.to_dict()["baseline_hits"] == 2
    assert any(expected in message.content for message in llm.calls[1]["messages"])


def validated_rule():
    return rules_module.validate_rule(RULE, RuleCollector(), run_semgrep_validate=False)


@pytest.fixture
def fake_scan(tmp_path, monkeypatch):
    for name in ("accepted.py", "changed.py", "file with spaces.py"):
        (tmp_path / name).write_text("pass\n")
    scans = []
    hits = []

    def run(cmd, **kwargs):
        assert kwargs["cwd"] == str(tmp_path)
        assert kwargs["capture_output"] and kwargs["text"]
        if cmd[0] == "git":
            assert cmd[1:] == ["ls-files", "-z"]
            return subprocess.CompletedProcess(
                cmd, 0, "accepted.py\0changed.py\0file with spaces.py\0", ""
            )
        assert cmd[0] == "semgrep"
        assert kwargs["timeout"] == 10
        config_path = cmd[cmd.index("--config") + 1]
        with open(config_path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        scans.append((cmd, config, config_path))
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"results": hits}), "")

    monkeypatch.setattr(subprocess, "run", run)
    return scans, hits


def test_quiet_rule_is_accepted_and_only_unchanged_tracked_files_are_scanned(
    tmp_path, fake_scan
):
    scans, _ = fake_scan
    rule = rules_module.check_rule_quietness(validated_rule(), str(tmp_path), ["changed.py"])

    assert rule.valid and rule.baseline_hits == 0
    assert rule.to_dict()["baseline_hits"] == 0
    cmd, config, config_path = scans[0]
    assert "--json" in cmd
    assert "accepted.py" in cmd and "file with spaces.py" in cmd
    assert "changed.py" not in cmd
    assert config["rules"][0]["id"] == "decorator-order"
    assert not os.path.exists(config_path)


def test_hits_on_changed_files_and_other_rule_ids_are_not_baseline_hits(tmp_path, fake_scan):
    _, hits = fake_scan
    hits.extend([
        {"check_id": "tmp.config.decorator-order", "path": "./changed.py", "start": {"line": 3}},
        {"check_id": "tmp.config.decorator-order", "path": str(tmp_path / "changed.py"),
         "start": {"line": 4}},
        {"check_id": "tmp.config.other-rule", "path": "accepted.py", "start": {"line": 5}},
    ])
    rule = rules_module.check_rule_quietness(validated_rule(), str(tmp_path), ["changed.py"])
    assert rule.valid and rule.baseline_hits == 0


def test_missing_semgrep_leaves_rule_valid(tmp_path, monkeypatch):
    def run(cmd, **kwargs):
        if cmd[0] == "git":
            return subprocess.CompletedProcess(cmd, 0, "accepted.py\0", "")
        raise FileNotFoundError("semgrep")

    monkeypatch.setattr(subprocess, "run", run)
    rule = rules_module.check_rule_quietness(validated_rule(), str(tmp_path), [])
    assert rule.valid and rule.baseline_hits == 0


def test_malformed_yaml_is_rejected_before_quietness_check(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid YAML must not reach a subprocess or quietness check")

    monkeypatch.setattr(subprocess, "run", unexpected)
    context = ReviewContext(str(tmp_path), DiffMap.from_pull_files({}))
    collector = RuleCollector()
    belt = Toolbelt(context, FindingCollector(), Budget(), rule_collector=collector)
    out = belt.dispatch(ToolCall(id="bad", name="write_rule", arguments={
        "rule_yaml": "id: [unclosed", "rationale": "test",
    }))
    assert "Rule rejected: invalid YAML" in out.text
    assert not collector.rules[0].valid
    assert collector.rules[0].baseline_hits == 0


@pytest.mark.skipif(shutil.which("semgrep") is None, reason="semgrep not installed")
def test_real_semgrep_rejects_decorator_rule_on_accepted_file(tmp_path):
    source = (
        "class Example:\n"
        "    @classmethod\n"
        "    @cache\n"
        "    def existing(cls):\n"
        "        return 1\n"
    )
    (tmp_path / "accepted.py").write_text(source)
    (tmp_path / "changed.py").write_text(source)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "accepted.py", "changed.py"], cwd=tmp_path, check=True)

    rule = rules_module.check_rule_quietness(validated_rule(), str(tmp_path), ["changed.py"])
    assert not rule.valid
    assert rule.baseline_hits == 1
    assert "accepted.py:2" in rule.rejected_because


def test_missing_git_skips_check_when_no_tracked_listing_is_available(tmp_path, monkeypatch):
    def run(cmd, **kwargs):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(subprocess, "run", run)
    rule = rules_module.check_rule_quietness(validated_rule(), str(tmp_path), [])
    assert rule.valid and rule.baseline_hits == 0


def test_no_unchanged_files_does_not_scan_the_workspace(tmp_path, monkeypatch):
    def run(cmd, **kwargs):
        assert cmd[0] == "git", "semgrep must not default to scanning the whole workspace"
        return subprocess.CompletedProcess(cmd, 0, "changed.py\0", "")

    monkeypatch.setattr(subprocess, "run", run)
    rule = rules_module.check_rule_quietness(validated_rule(), str(tmp_path), ["changed.py"])
    assert rule.valid and rule.baseline_hits == 0


@pytest.mark.parametrize("failure", ["timeout", "bad-json"])
def test_semgrep_failure_skips_check(tmp_path, monkeypatch, failure):
    def run(cmd, **kwargs):
        if cmd[0] == "git":
            return subprocess.CompletedProcess(cmd, 0, "accepted.py\0", "")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(cmd, 10)
        return subprocess.CompletedProcess(cmd, 0, "not json", "")

    monkeypatch.setattr(subprocess, "run", run)
    rule = rules_module.check_rule_quietness(validated_rule(), str(tmp_path), [])
    assert rule.valid and rule.baseline_hits == 0


def test_disabling_semgrep_returns_rule_unchanged(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("disabled quietness check must not invoke subprocesses")

    monkeypatch.setattr(subprocess, "run", unexpected)
    rule = rules_module.check_rule_quietness(
        validated_rule(), str(tmp_path), [], run_semgrep=False
    )
    assert rule.valid and rule.baseline_hits == 0


def test_bad_rule_never_raises(tmp_path):
    rule = rules_module.CustomRule(rule_id="bad-rule", yaml_text="id: [unclosed", valid=True)
    result = rules_module.check_rule_quietness(rule, str(tmp_path), [])
    assert not result.valid and "invalid YAML" in result.rejected_because
