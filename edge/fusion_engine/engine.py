"""Fusion / Rule Engine — Phase 3.

RuleEngine: loads rules.yaml, evaluates against FusionInput dict.
FusionEngine: orchestrates rule evaluation + monitor events, builds Alarm.
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

ALARM_LEVELS = {"INFO": 0, "WARNING": 1, "HIGH": 2, "CRITICAL": 3}
VALID_OPERATORS = {"gt", "lt", "eq", "neq", "gte", "lte"}
VALID_ALARM_LEVELS = set(ALARM_LEVELS.keys())


class RuleEngine:
    def __init__(self, rules_path: str) -> None:
        self.rules_path = Path(rules_path)
        self.rules: list[dict] = []
        self._load_and_validate()

    def _load_and_validate(self) -> None:
        with open(self.rules_path) as f:
            data = yaml.safe_load(f)
        rules = data.get("rules", [])
        for rule in rules:
            self._validate_rule(rule)
        self.rules = rules
        log.info("Loaded %d rules from %s", len(self.rules), self.rules_path)

    def _validate_rule(self, rule: dict) -> None:
        for field in ("id", "name", "conditions", "action"):
            if field not in rule:
                raise ValueError(f"Rule missing field '{field}': {rule}")
        action = rule["action"]
        for af in ("alarm_level", "message_key"):
            if af not in action:
                raise ValueError(f"Rule {rule['id']} action missing '{af}'")
        if action["alarm_level"] not in VALID_ALARM_LEVELS:
            raise ValueError(
                f"Rule {rule['id']} invalid alarm_level: {action['alarm_level']}"
            )
        for cond in rule["conditions"]:
            for cf in ("field", "operator", "value"):
                if cf not in cond:
                    raise ValueError(
                        f"Rule {rule['id']} condition missing '{cf}': {cond}"
                    )
            if cond["operator"] not in VALID_OPERATORS:
                raise ValueError(
                    f"Rule {rule['id']} unknown operator: {cond['operator']}"
                )

    def reload_rules(self) -> None:
        self._load_and_validate()

    def evaluate(self, fusion_input: dict) -> dict:
        for rule in self.rules:
            if self._check_conditions(rule["conditions"], fusion_input):
                return {
                    "matched_rule": rule["id"],
                    "alarm_level": rule["action"]["alarm_level"],
                    "message_key": rule["action"]["message_key"],
                    "source": "rule_engine",
                }
        return {
            "matched_rule": None,
            "alarm_level": "INFO",
            "message_key": "no_rule_matched",
            "source": "rule_engine",
        }

    def _check_conditions(self, conditions: list, data: dict) -> bool:
        return all(self._evaluate_condition(c, data) for c in conditions)

    def _evaluate_condition(self, condition: dict, data: dict) -> bool:
        value = self._get_nested(data, condition["field"])
        if value is None:
            return False
        op = condition["operator"]
        threshold = condition["value"]
        ops = {
            "gt":  lambda a, b: a > b,
            "lt":  lambda a, b: a < b,
            "eq":  lambda a, b: a == b,
            "neq": lambda a, b: a != b,
            "gte": lambda a, b: a >= b,
            "lte": lambda a, b: a <= b,
        }
        if op not in ops:
            raise ValueError(f"Unknown operator: {op}")
        return ops[op](value, threshold)

    def _get_nested(self, data: dict, field_path: str) -> Any:
        keys = field_path.split(".")
        val = data
        try:
            for k in keys:
                val = val[k]
            return val
        except (KeyError, TypeError):
            log.warning("Field path '%s' not found in input — condition evaluates False", field_path)
            return None

    def get_rules(self) -> list[dict]:
        return self.rules

    def update_threshold(self, rule_id: str, field: str, value: float) -> dict:
        for rule in self.rules:
            if rule["id"] == rule_id:
                for cond in rule["conditions"]:
                    if cond["field"] == field:
                        cond["value"] = value
                        self._persist_rules()
                        return rule
                raise ValueError(f"Field '{field}' not found in rule {rule_id}")
        raise ValueError(f"Rule '{rule_id}' not found")

    def _persist_rules(self) -> None:
        with open(self.rules_path, "r") as f:
            raw = yaml.safe_load(f)
        raw["rules"] = self.rules
        with open(self.rules_path, "w") as f:
            yaml.dump(raw, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        log.info("rules.yaml updated")


class FusionEngine:
    def __init__(self, rules_path: str) -> None:
        self.rule_engine = RuleEngine(rules_path)

    def process(self, fusion_input: dict) -> dict:
        rule_result = self.rule_engine.evaluate(fusion_input)
        monitor_result = self._evaluate_monitor_events(
            fusion_input.get("monitor_events", [])
        )
        decision = self._merge_results(rule_result, monitor_result)
        return self._build_alarm(fusion_input["node_id"], decision, fusion_input)

    def _evaluate_monitor_events(self, events: list) -> dict:
        anomalous = [e for e in events if e.get("is_anomaly")]
        if not anomalous:
            return {
                "alarm_level": "INFO",
                "source": "model_monitor",
                "matched_rule": None,
                "message_key": "monitor_normal",
            }
        worst = max(anomalous, key=lambda e: abs(e.get("z_score", 0)))
        return {
            "alarm_level": "WARNING",
            "source": "model_monitor",
            "matched_rule": None,
            "message_key": "model_output_drift",
            "trigger_model": worst.get("model_name"),
            "z_score": worst.get("z_score"),
        }

    def _merge_results(self, rule_result: dict, monitor_result: dict) -> dict:
        if (
            ALARM_LEVELS[monitor_result["alarm_level"]]
            > ALARM_LEVELS[rule_result["alarm_level"]]
        ):
            return monitor_result
        return rule_result

    def _build_alarm(self, node_id: str, decision: dict, fusion_input: dict) -> dict:
        alarm_id = f"ALM_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6].upper()}"
        return {
            "alarm_id": alarm_id,
            "node_id": node_id,
            "timestamp": fusion_input["timestamp"],
            "level": decision["alarm_level"],
            "source": decision["source"],
            "rule_id": decision.get("matched_rule"),
            "trigger_values": self._extract_trigger_values(decision, fusion_input),
            "llm_explanation": None,
            "synced": False,
        }

    def _extract_trigger_values(self, decision: dict, fusion_input: dict) -> dict:
        trigger: dict = {}
        matched_rule = decision.get("matched_rule")
        if matched_rule:
            for rule in self.rule_engine.rules:
                if rule["id"] == matched_rule:
                    for cond in rule["conditions"]:
                        val = self.rule_engine._get_nested(fusion_input, cond["field"])
                        trigger[cond["field"]] = val
                    break
        if decision.get("trigger_model"):
            trigger["trigger_model"] = decision["trigger_model"]
            trigger["z_score"] = decision.get("z_score")
        return trigger
