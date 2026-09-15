#!/usr/bin/env python3
"""Grab scarce on-demand SageMaker GPU capacity in one region, then hold it.

The same file runs four ways:

    python3 src/sm_capacity_hunter.py --preflight   # checks only, no writes
    python3 src/sm_capacity_hunter.py --status      # read-only: what is the state right now
    python3 src/sm_capacity_hunter.py --once        # advance the state machine exactly one step
    python3 src/sm_capacity_hunter.py --loop        # long-running hunter + watchdog

    lambda_handler(event, context)                  # one tick per invocation, via the SAM stack

CLI flags override environment variables, so the same module serves both. The
SAM template in ../template.yaml supplies the environment for the Lambda path.

Design note that makes the Lambda mode work: the live endpoint IS the state
store.  Every tick reads DescribeEndpoint and advances exactly one step, so
Lambda's 15-minute ceiling is irrelevant, nothing needs DynamoDB, and two
runners racing each other cost a wasted API call at worst.

State machine, one step per tick:

    absent                  -> ensure model+config, CreateEndpoint        HUNTING
    Creating/Deleting/...   -> nothing, SageMaker is working              HUNTING
    Failed  (capacity)      -> DeleteEndpoint, so the next tick re-creates HUNTING
    Failed  (fatal)         -> stop, notify, leave it Failed for a human  HALTED
    Failed  (unknown)       -> HALTED unless --retry-unknown              HALTED
    OutOfService            -> wait out the grace period, then re-hunt    DEGRADED
    InService, right type   -> nothing.  This is the goal.                HOLDING
    InService, wrong type   -> nothing, but say so loudly                 HOLDING_DEGRADED

Why "Failed (fatal) -> stop" matters: a container that fails its ping health
check, a missing model artefact and a zero service quota all produce a Failed
endpoint that looks exactly like scarcity if you only count attempts.  Retrying
those forever is a silent no-op that reads as bad luck.  So failure reasons are
classified, and anything not recognisably a capacity error halts the hunt and
tells you.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

REGION_DEFAULT = "ap-southeast-2"
INSTANCE_TYPE_DEFAULT = "ml.g6.2xlarge"
ENDPOINT_NAME_DEFAULT = "capacity-hold-ep"

# EndpointStatus values from the SageMaker API.
TRANSITIONAL = {"Creating", "Updating", "SystemUpdating", "RollingBack", "Deleting"}
HALT_STATES = {"UpdateRollbackFailed"}


# --------------------------------------------------------------------------- logs


def log(event: str, **fields) -> None:
    """One JSON object per line: greppable locally, queryable in CloudWatch Logs."""
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": event}
    rec.update(fields)
    print(json.dumps(rec, default=str), flush=True)


# ------------------------------------------------------------------- classification

# Checked first.  These mean "the pool is empty, come back later" -> keep hunting.
CAPACITY_RE = re.compile(
    r"capacityerror"
    r"|insufficient\s+(instance\s+)?capacity"
    r"|not\s+enough\s+capacity"
    r"|unable\s+to\s+(provision|satisfy)"
    r"|could\s+not\s+provision"
    r"|capacity\s+is\s+(currently\s+)?(un)?available"
    r"|retry\s+using\s+a\s+different"
    r"|please\s+(try|retry)\s+again",
    re.I,
)

# Checked second.  These mean "your configuration is wrong" -> retrying is a no-op.
FATAL_RE = re.compile(
    r"resourcelimitexceeded"
    r"|account[-\s]level\s+service\s+limit"
    r"|exceeded\s+the\s+\S+\s+limit"
    r"|ping\s+health\s+check"
    r"|did\s+not\s+pass"
    r"|access\s*denied|not\s+authorized"
    r"|no\s+such\s+key|could\s+not\s+find\s+model\s+data"
    r"|(cannot|could\s+not)\s+be\s+found|does\s+not\s+exist"
    r"|(pull|download)\S*\s+.{0,40}image|imagepull|manifest\s+.{0,20}not\s+found"
    r"|invalid",
    re.I,
)


def classify(reason: str) -> str:
    """-> 'capacity' | 'fatal' | 'unknown'.  Order matters: capacity wins."""
    if not reason:
        return "unknown"
    if CAPACITY_RE.search(reason):
        return "capacity"
    if FATAL_RE.search(reason):
        return "fatal"
    return "unknown"


# ------------------------------------------------------------------------- config


def _env(name, default=None):
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _env_int(name, default):
    v = _env(name)
    return int(v) if v is not None else default


def _env_bool(name, default=False):
    v = _env(name)
    return v.strip().lower() in ("1", "true", "yes", "on") if v is not None else default


def _env_json(name, default):
    v = _env(name)
    if v is None:
        return default
    try:
        return json.loads(v)
    except json.JSONDecodeError as e:
        # A malformed CONTAINER_ENV must not degrade into "no environment".
        raise SystemExit(f"{name} is not valid JSON ({e}): {v!r}") from e


def _env_list(name):
    v = _env(name)
    return [x.strip() for x in v.split(",") if x.strip()] if v else []


class Cfg:
    """Config from environment (Lambda) with argparse overriding it (CLI)."""

    def __init__(self, overrides: dict | None = None):
        o = {k: v for k, v in (overrides or {}).items() if v is not None}

        self.region = o.get("region") or _env("REGION", REGION_DEFAULT)
        self.instance_type = o.get("instance_type") or _env("INSTANCE_TYPE", INSTANCE_TYPE_DEFAULT)
        self.count = o.get("count") or _env_int("INSTANCE_COUNT", 1)
        self.endpoint_name = o.get("endpoint_name") or _env("ENDPOINT_NAME", ENDPOINT_NAME_DEFAULT)

        self.model_name = o.get("model_name") or _env("MODEL_NAME") or f"{self.endpoint_name}-model"
        self.image_uri = o.get("image_uri") or _env("IMAGE_URI")
        self.model_data_url = o.get("model_data_url") or _env("MODEL_DATA_URL")
        self.role_arn = o.get("role_arn") or _env("ROLE_ARN")
        self.container_env = o.get("container_env") or _env_json("CONTAINER_ENV", {})
        self.subnet_ids = o.get("subnet_ids") or _env_list("SUBNET_IDS")
        self.security_group_ids = o.get("security_group_ids") or _env_list("SECURITY_GROUP_IDS")

        self.sns_topic_arn = o.get("sns_topic_arn") or _env("SNS_TOPIC_ARN")
        self.retry_unknown = o.get("retry_unknown") or _env_bool("RETRY_UNKNOWN", False)
        self.reclaim_after = o.get("reclaim_after")
        if self.reclaim_after is None:
            self.reclaim_after = _env_int("RECLAIM_AFTER_SECONDS", 900)
        if self.reclaim_after is not None and self.reclaim_after < 0:
            self.reclaim_after = None  # never auto-delete an OutOfService endpoint

        self.hunt_poll = o.get("hunt_poll") or _env_int("HUNT_POLL_SECONDS", 20)
        self.hold_poll = o.get("hold_poll") or _env_int("HOLD_POLL_SECONDS", 60)
        # Stateless notify de-dupe: only alert on a *fresh* transition.
        self.notify_window = o.get("notify_window") or _env_int("NOTIFY_WINDOW_SECONDS", 180)
        self.dry_run = bool(o.get("dry_run") or _env_bool("DRY_RUN", False))

        self.tags = o.get("tags") or _env_json(
            "TAGS",
            [
                {"Key": "managed-by", "Value": "sm-capacity-hunter"},
                {"Key": "purpose", "Value": "on-demand-capacity-hold"},
            ],
        )

    @property
    def config_name(self) -> str:
        name = f"{self.endpoint_name}-{self.instance_type.replace('.', '-')}-x{self.count}"
        return name[:63].rstrip("-")

    def summary(self) -> dict:
        return {
            "region": self.region,
            "endpoint": self.endpoint_name,
            "instance_type": self.instance_type,
            "count": self.count,
            "endpoint_config": self.config_name,
            "model": self.model_name,
            "retry_unknown": self.retry_unknown,
            "reclaim_after_s": self.reclaim_after,
            "dry_run": self.dry_run,
        }


# --------------------------------------------------------------------- aws plumbing

RETRYABLE_API = {
    "ThrottlingException",
    "Throttling",
    "ThrottledException",
    "TooManyRequestsException",
    "RequestLimitExceeded",
    "RequestThrottled",
    "ServiceUnavailable",
    "InternalFailure",
    "InternalServerError",
    "InternalError",
}


def clients(c: Cfg):
    cfg = BotoConfig(
        region_name=c.region,
        retries={"max_attempts": 3, "mode": "standard"},
        user_agent_extra="sm-capacity-hunter/1.0",
    )
    sm = boto3.client("sagemaker", config=cfg)
    sq = boto3.client("service-quotas", config=cfg)
    sns = boto3.client("sns", config=cfg) if c.sns_topic_arn else None
    return sm, sq, sns


def call(fn, _attempts: int = 6, **kw):
    """Invoke a boto3 operation, retrying transient API errors only."""
    for attempt in range(1, _attempts + 1):
        try:
            return fn(**kw)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code not in RETRYABLE_API or attempt == _attempts:
                raise
            nap = min(30.0, (2 ** (attempt - 1)) * (0.5 + random.random()))
            log("api_retry", op=getattr(fn, "__name__", "?"), code=code, attempt=attempt, sleep_s=round(nap, 2))
            time.sleep(nap)


def _missing(e: ClientError) -> bool:
    err = e.response.get("Error", {})
    msg = (err.get("Message") or "").lower()
    return err.get("Code") in ("ValidationException", "ResourceNotFound") and (
        "not find" in msg or "does not exist" in msg or "not found" in msg
    )


def _age_s(ts) -> float | None:
    if not ts:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return round((datetime.now(timezone.utc) - ts).total_seconds(), 1)


def describe_endpoint(sm, name: str) -> dict | None:
    try:
        return call(sm.describe_endpoint, EndpointName=name)
    except ClientError as e:
        if _missing(e):
            return None
        raise


def notify(sns, c: Cfg, subject: str, body: dict) -> None:
    if not sns or not c.sns_topic_arn:
        return
    try:
        call(
            sns.publish,
            TopicArn=c.sns_topic_arn,
            Subject=f"[capacity-hunter] {subject}: {c.endpoint_name}"[:100],
            Message=json.dumps({"subject": subject, **c.summary(), **body}, indent=2, default=str),
        )
        log("notified", subject=subject)
    except (ClientError, BotoCoreError) as e:
        # A failed page must never take down the hunt, but it must be visible.
        log("notify_failed", subject=subject, error=str(e))


# ------------------------------------------------------------- model / config setup


def ensure_model(sm, c: Cfg) -> str:
    try:
        call(sm.describe_model, ModelName=c.model_name)
        return c.model_name
    except ClientError as e:
        if not _missing(e):
            raise

    if not (c.image_uri and c.role_arn):
        raise SystemExit(
            f"model {c.model_name!r} does not exist in {c.region} and IMAGE_URI/ROLE_ARN "
            "were not supplied, so it cannot be created"
        )

    container = {"Image": c.image_uri}
    if c.model_data_url:
        container["ModelDataUrl"] = c.model_data_url
    if c.container_env:
        container["Environment"] = {k: str(v) for k, v in c.container_env.items()}

    kw = {"ModelName": c.model_name, "PrimaryContainer": container, "ExecutionRoleArn": c.role_arn}
    if c.subnet_ids and c.security_group_ids:
        kw["VpcConfig"] = {"Subnets": c.subnet_ids, "SecurityGroupIds": c.security_group_ids}
    if c.tags:
        kw["Tags"] = c.tags

    call(sm.create_model, **kw)
    call(sm.describe_model, ModelName=c.model_name)  # post-condition, not the CreateModel return
    log("model_created", model=c.model_name, image=c.image_uri)
    return c.model_name


def ensure_endpoint_config(sm, c: Cfg) -> str:
    name = c.config_name
    try:
        d = call(sm.describe_endpoint_config, EndpointConfigName=name)
        v = (d.get("ProductionVariants") or [{}])[0]
        if v.get("InstanceType") != c.instance_type or v.get("InitialInstanceCount") != c.count:
            raise SystemExit(
                f"endpoint config {name!r} already exists but specifies "
                f"{v.get('InstanceType')} x{v.get('InitialInstanceCount')}, "
                f"not {c.instance_type} x{c.count}. Delete it or pick a new --endpoint-name."
            )
        return name
    except ClientError as e:
        if not _missing(e):
            raise

    ensure_model(sm, c)
    kw = {
        "EndpointConfigName": name,
        "ProductionVariants": [
            {
                "VariantName": "AllTraffic",
                "ModelName": c.model_name,
                "InstanceType": c.instance_type,
                "InitialInstanceCount": c.count,
                "InitialVariantWeight": 1.0,
            }
        ],
    }
    if c.tags:
        kw["Tags"] = c.tags
    call(sm.create_endpoint_config, **kw)
    call(sm.describe_endpoint_config, EndpointConfigName=name)  # post-condition
    log("endpoint_config_created", config=name, instance_type=c.instance_type, count=c.count)
    return name


# ---------------------------------------------------------------------- state machine


def _result(state: str, action: str, **fields) -> dict:
    return {"state": state, "action": action, **fields}


def tick(sm, c: Cfg, sns=None, act: bool = True) -> dict:
    """Advance the hunt by exactly one step.  act=False makes it read-only."""
    ep = describe_endpoint(sm, c.endpoint_name)
    write = act and not c.dry_run

    # ---- nothing there: try to grab capacity.
    if ep is None:
        if not write:
            return _result("hunting", "would_create_endpoint", endpoint_exists=False)
        cfg_name = ensure_endpoint_config(sm, c)
        try:
            call(sm.create_endpoint, EndpointName=c.endpoint_name, EndpointConfigName=cfg_name, Tags=c.tags)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            msg = e.response.get("Error", {}).get("Message", "")
            if code == "ResourceLimitExceeded":
                res = _result("halted", "none", failure_kind="fatal", reason=f"{code}: {msg}",
                              hint="this is a service quota, not scarcity — retrying cannot succeed")
                notify(sns, c, "HALTED (quota)", res)
                return res
            if code == "ResourceInUse":
                return _result("hunting", "wait", note="endpoint appeared between describe and create")
            raise
        # Do not trust the CreateEndpoint return: read the endpoint back.
        after = describe_endpoint(sm, c.endpoint_name)
        observed = (after or {}).get("EndpointStatus")
        if observed != "Creating":
            log("post_condition_warning", op="create_endpoint", expected="Creating", observed=observed)
        return _result("hunting", "create_endpoint", status=observed, endpoint_config=cfg_name)

    status = ep.get("EndpointStatus", "")
    age = _age_s(ep.get("LastModifiedTime"))
    fresh = age is not None and age < c.notify_window

    # ---- holding.  Verify the artefact, do not infer it from the status word.
    if status == "InService":
        variant = (ep.get("ProductionVariants") or [{}])[0]
        current = variant.get("CurrentInstanceCount")
        held_type = None
        try:
            live = call(sm.describe_endpoint_config, EndpointConfigName=ep.get("EndpointConfigName", ""))
            held_type = (live.get("ProductionVariants") or [{}])[0].get("InstanceType")
        except ClientError as e:
            log("config_read_failed", error=str(e))  # held_type stays None = UNKNOWN, not OK

        on_target = held_type == c.instance_type and isinstance(current, int) and current >= c.count
        res = _result(
            "holding" if on_target else "holding_degraded",
            "none",
            held_instance_type=held_type,
            current_instance_count=current,
            desired_instance_count=variant.get("DesiredInstanceCount"),
            wanted=f"{c.instance_type} x{c.count}",
            endpoint_arn=ep.get("EndpointArn"),
            in_service_for_s=age,
        )
        if fresh:
            notify(sns, c, "ACQUIRED" if on_target else "ACQUIRED (wrong shape)", res)
        return res

    # ---- SageMaker is mid-flight; leave it alone.
    if status in TRANSITIONAL:
        return _result("hunting", "wait", status=status, status_age_s=age)

    # ---- the interesting case.
    if status == "Failed":
        reason = ep.get("FailureReason") or ""
        kind = classify(reason)
        if kind == "capacity" or (kind == "unknown" and c.retry_unknown):
            if not write:
                return _result("hunting", "would_delete_failed", failure_kind=kind, reason=reason)
            call(sm.delete_endpoint, EndpointName=c.endpoint_name)
            after = describe_endpoint(sm, c.endpoint_name)
            observed = (after or {}).get("EndpointStatus")
            if observed not in (None, "Deleting"):
                log("post_condition_warning", op="delete_endpoint", expected="Deleting|absent", observed=observed)
            return _result("hunting", "delete_failed", failure_kind=kind, reason=reason,
                           attempt_took_s=age, status_after=observed)
        res = _result("halted", "none", failure_kind=kind, reason=reason,
                      hint="not a recognised capacity error — fix the cause, or pass --retry-unknown "
                           "if you are sure it is transient")
        if fresh:
            notify(sns, c, f"HALTED ({kind})", res)
        return res

    # ---- had capacity, lost it.  This is what the watchdog exists for.
    if status == "OutOfService":
        if c.reclaim_after is not None and age is not None and age >= c.reclaim_after:
            if not write:
                return _result("hunting", "would_delete_out_of_service", out_of_service_for_s=age)
            call(sm.delete_endpoint, EndpointName=c.endpoint_name)
            return _result("hunting", "delete_out_of_service", out_of_service_for_s=age,
                           note="re-hunting; capacity may not come back immediately")
        res = _result("degraded", "wait", status=status, out_of_service_for_s=age,
                      reclaim_after_s=c.reclaim_after)
        if fresh:
            notify(sns, c, "LOST CAPACITY", res)
        return res

    if status in HALT_STATES:
        res = _result("halted", "none", status=status, reason=ep.get("FailureReason") or "",
                      hint="needs a human: rollback did not complete")
        if fresh:
            notify(sns, c, f"HALTED ({status})", res)
        return res

    return _result("halted", "none", status=status, reason="unrecognised EndpointStatus")


# -------------------------------------------------------------------------- preflight


def preflight(c: Cfg, sm, sq) -> list[dict]:
    """OK / UNKNOWN / FAIL per check.  UNKNOWN is never reported as OK."""
    out: list[dict] = []

    def add(level, check, detail):
        out.append({"level": level, "check": check, "detail": detail})

    add("OK", "target", f"{c.instance_type} x{c.count} in {c.region} as endpoint {c.endpoint_name!r}")

    # Where this code runs versus where the endpoint goes. Differing is legal and
    # sometimes deliberate, but an unnoticed divergence sends you hunting for the
    # endpoint in the wrong console region.
    caller = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if caller and caller != c.region:
        add("OK", "region", f"hunting in {c.region} while this code runs in {caller} — "
                            "cross-region is supported, just confirm it is intended")
    elif caller:
        add("OK", "region", f"running and hunting in {c.region}")
    else:
        add("OK", "region", f"hunting in {c.region}; no caller region set in the environment")

    # 1. Service quota. Zero quota fails identically to scarcity, so check it first.
    want = f"{c.instance_type} for endpoint usage"
    quota = None
    source = None
    try:
        for op, kind in (("list_service_quotas", "applied"), ("list_aws_default_service_quotas", "default")):
            for page in sq.get_paginator(op).paginate(ServiceCode="sagemaker"):
                for q in page.get("Quotas", []):
                    if (q.get("QuotaName") or "").strip().lower() == want.lower():
                        quota, source = q, kind
                        break
                if quota:
                    break
            if quota:
                break
        if quota is None:
            add("UNKNOWN", "quota",
                f"no quota named {want!r} came back for service 'sagemaker' in {c.region}. "
                "Could not read it — that is not evidence the quota is fine.")
        elif (quota.get("Value") or 0) < c.count:
            add("FAIL", "quota",
                f"{want} = {quota.get('Value'):g} ({source}), need >= {c.count}. Raise this first: "
                "with a zero quota the hunt can never succeed and every attempt looks like scarcity.")
        else:
            add("OK", "quota", f"{want} = {quota.get('Value'):g} ({source})")
    except (ClientError, BotoCoreError) as e:
        add("UNKNOWN", "quota", f"Service Quotas unreadable ({e}) — quota state UNKNOWN, not OK")

    # 2. Container image must live in the same region as the endpoint.
    if c.image_uri:
        m = re.search(r"\.dkr\.ecr\.([a-z0-9-]+)\.amazonaws\.com", c.image_uri)
        if m and m.group(1) != c.region:
            add("FAIL", "image_region", f"image is in {m.group(1)} but the endpoint is in {c.region}")
        else:
            add("OK", "image_region", c.image_uri)
    else:
        add("OK", "image_region", "no IMAGE_URI given; relying on an existing model")

    # 3. Model: exists, or can be created.
    try:
        d = call(sm.describe_model, ModelName=c.model_name)
        add("OK", "model", f"{c.model_name} exists (image {d.get('PrimaryContainer', {}).get('Image')})")
    except ClientError as e:
        if not _missing(e):
            add("UNKNOWN", "model", f"could not describe {c.model_name}: {e}")
        elif c.image_uri and c.role_arn:
            add("OK", "model", f"{c.model_name} absent; will be created from {c.image_uri}")
        else:
            add("FAIL", "model", f"{c.model_name} absent and no IMAGE_URI/ROLE_ARN to create it")

    # 4. Existing endpoint state.
    try:
        ep = describe_endpoint(sm, c.endpoint_name)
        if ep is None:
            add("OK", "endpoint", "absent; the first tick will create it")
        else:
            add("OK", "endpoint",
                f"{ep.get('EndpointStatus')} (last modified {_age_s(ep.get('LastModifiedTime'))}s ago)"
                + (f" reason={ep.get('FailureReason')}" if ep.get("FailureReason") else ""))
    except (ClientError, BotoCoreError) as e:
        add("UNKNOWN", "endpoint", f"could not describe {c.endpoint_name}: {e}")

    # 5. The one thing that cannot be checked from here.
    add("OK", "container_health",
        "unverified by design: if the container fails its /ping health check the endpoint goes "
        "Failed for a reason classify() calls fatal, and the hunt halts instead of looping forever")
    return out


def print_preflight(findings: list[dict]) -> int:
    order = {"FAIL": 0, "UNKNOWN": 1, "OK": 2}
    for f in sorted(findings, key=lambda x: order.get(x["level"], 3)):
        print(f"  {f['level']:<8} {f['check']:<17} {f['detail']}")
    fails = sum(1 for f in findings if f["level"] == "FAIL")
    unknown = sum(1 for f in findings if f["level"] == "UNKNOWN")
    print(f"\n  {fails} FAIL, {unknown} UNKNOWN, {len(findings) - fails - unknown} OK")
    return fails


# ------------------------------------------------------------------------- run modes


def run_loop(c: Cfg, sm, sns) -> int:
    log("loop_start", **c.summary())
    consecutive_errors = 0
    last_state = None
    while True:
        try:
            res = tick(sm, c, sns)
            consecutive_errors = 0
        except (ClientError, BotoCoreError) as e:
            consecutive_errors += 1
            log("tick_error", error=str(e), consecutive=consecutive_errors)
            if consecutive_errors >= 20:
                log("loop_abort", reason="20 consecutive API errors")
                return 3
            time.sleep(min(300, 5 * consecutive_errors))
            continue
        except SystemExit as e:  # configuration problem raised by ensure_*
            log("loop_abort", reason=str(e))
            return 2

        if res["state"] != last_state:
            log("state_change", **{"from": last_state}, **res)
            last_state = res["state"]
        else:
            log("tick", **res)

        if res["state"] == "halted":
            log("loop_abort", reason="halted", **res)
            return 2
        time.sleep(c.hold_poll if res["state"].startswith("holding") else c.hunt_poll)


def release(c: Cfg, sm) -> int:
    ep = describe_endpoint(sm, c.endpoint_name)
    if ep is None:
        log("release", result="already absent", endpoint=c.endpoint_name)
        return 0
    log("release", endpoint=c.endpoint_name, status=ep.get("EndpointStatus"),
        warning="this frees the instances back to the shared pool; getting them again may take days")
    call(sm.delete_endpoint, EndpointName=c.endpoint_name)
    for _ in range(60):
        time.sleep(5)
        if describe_endpoint(sm, c.endpoint_name) is None:
            log("release", result="deleted", endpoint=c.endpoint_name)
            return 0
    log("release", result="still deleting after 300s — verify manually", endpoint=c.endpoint_name)
    return 1


# ---------------------------------------------------------------------------- lambda


def lambda_handler(event, context):  # noqa: ARG001
    """One tick per invocation. Schedule with EventBridge at rate(1 minute).

    Exceptions deliberately propagate: a thrown Lambda shows up in Errors and
    the next scheduled tick retries anyway. Swallowing them would make a broken
    hunter look like a patient one.
    """
    c = Cfg()
    sm, sq, sns = clients(c)
    if isinstance(event, dict) and event.get("preflight"):
        findings = preflight(c, sm, sq)
        log("preflight", findings=findings)
        return {"findings": findings,
                "fail": [f for f in findings if f["level"] == "FAIL"],
                "config": c.summary()}
    res = tick(sm, c, sns)
    log("tick", **res, **{"endpoint": c.endpoint_name, "region": c.region})
    return res


# ------------------------------------------------------------------------------ cli


def parse_args(argv=None):
    # No ArgumentDefaultsHelpFormatter: every flag defaults to None here so that
    # Cfg can tell "not passed" from "passed", and the real default comes from the
    # environment or a module constant. The formatter would print "(default: None)"
    # against flags that do have defaults, which is worse than saying nothing.
    p = argparse.ArgumentParser(
        description="Hunt scarce on-demand SageMaker GPU capacity and hold it. "
                    "Unset flags fall back to the matching environment variable, "
                    "then to the default shown below.",
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--loop", action="store_true", help="run until it holds capacity, then keep watch")
    mode.add_argument("--once", action="store_true", help="advance the state machine one step and exit")
    mode.add_argument("--status", action="store_true", help="read-only: report state, change nothing")
    mode.add_argument("--preflight", action="store_true", help="checks only, no writes")
    mode.add_argument("--release", action="store_true", help="delete the endpoint and free the capacity")

    p.add_argument("--region", default=None,
                   help=f"region to hunt in (env REGION, else {REGION_DEFAULT})")
    p.add_argument("--instance-type", default=None,
                   help=f"env INSTANCE_TYPE, else {INSTANCE_TYPE_DEFAULT}. Also ml.g6.4xlarge etc")
    p.add_argument("--count", type=int, default=None,
                   help="instances to hold (env INSTANCE_COUNT, else 1)")
    p.add_argument("--endpoint-name", default=None,
                   help=f"env ENDPOINT_NAME, else {ENDPOINT_NAME_DEFAULT}")
    p.add_argument("--model-name", default=None,
                   help="env MODEL_NAME, else <endpoint-name>-model")
    p.add_argument("--image-uri", default=None,
                   help="inference container; must be in the same region as the endpoint (env IMAGE_URI)")
    p.add_argument("--model-data-url", default=None,
                   help="s3:// model.tar.gz, if the image needs one (env MODEL_DATA_URL)")
    p.add_argument("--role-arn", default=None, help="SageMaker execution role (env ROLE_ARN)")
    p.add_argument("--subnet-ids", default=None, help="comma separated; needs --security-group-ids too")
    p.add_argument("--security-group-ids", default=None, help="comma separated; needs --subnet-ids too")
    p.add_argument("--sns-topic-arn", default=None,
                   help="notify on acquire / halt / capacity loss (env SNS_TOPIC_ARN)")
    p.add_argument("--retry-unknown", action="store_true", default=None,
                   help="treat unrecognised FailureReasons as retryable; default is to halt and tell you")
    p.add_argument("--reclaim-after", type=int, default=None,
                   help="seconds OutOfService before delete+re-hunt; negative disables (default 900)")
    p.add_argument("--hunt-poll", type=int, default=None,
                   help="--loop seconds between ticks while hunting (default 20)")
    p.add_argument("--hold-poll", type=int, default=None,
                   help="--loop seconds between ticks while holding (default 60)")
    p.add_argument("--dry-run", action="store_true", default=None,
                   help="report what each tick would do without writing anything")
    p.add_argument("--strict", action="store_true", help="treat preflight UNKNOWN as failure")
    p.add_argument("--yes", action="store_true", help="required by --release")
    p.add_argument("--skip-preflight", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    overrides = {
        "region": a.region,
        "instance_type": a.instance_type,
        "count": a.count,
        "endpoint_name": a.endpoint_name,
        "model_name": a.model_name,
        "image_uri": a.image_uri,
        "model_data_url": a.model_data_url,
        "role_arn": a.role_arn,
        "subnet_ids": [s.strip() for s in a.subnet_ids.split(",")] if a.subnet_ids else None,
        "security_group_ids": [s.strip() for s in a.security_group_ids.split(",")] if a.security_group_ids else None,
        "sns_topic_arn": a.sns_topic_arn,
        "retry_unknown": a.retry_unknown,
        "reclaim_after": a.reclaim_after,
        "hunt_poll": a.hunt_poll,
        "hold_poll": a.hold_poll,
        "dry_run": a.dry_run,
    }
    c = Cfg(overrides)
    sm, sq, sns = clients(c)

    if a.release:
        if not a.yes:
            print(f"--release deletes endpoint {c.endpoint_name!r} in {c.region} and hands the "
                  f"{c.instance_type} back to the shared pool. Re-run with --yes to confirm.")
            return 1
        return release(c, sm)

    if a.preflight or (a.loop and not a.skip_preflight):
        print(f"\npreflight — {c.instance_type} x{c.count} in {c.region}\n")
        findings = preflight(c, sm, sq)
        fails = print_preflight(findings)
        print()
        unknown = sum(1 for f in findings if f["level"] == "UNKNOWN")
        if fails or (a.strict and unknown):
            return 1
        if a.preflight:
            return 0

    if a.status:
        res = tick(sm, c, sns, act=False)
        log("status", **res, **{"endpoint": c.endpoint_name, "region": c.region})
        return 0

    if a.once:
        res = tick(sm, c, sns)
        log("tick", **res, **{"endpoint": c.endpoint_name, "region": c.region})
        return 2 if res["state"] == "halted" else 0

    return run_loop(c, sm, sns)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        # Never delete anything on the way out: the endpoint may be holding capacity.
        log("interrupted", note="endpoint left exactly as it is; re-run to resume the hunt")
        sys.exit(130)
