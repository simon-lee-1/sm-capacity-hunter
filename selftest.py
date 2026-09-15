#!/usr/bin/env python3
"""Credential-free test of the state machine, the failure classifier, and the
wiring between template.yaml and the code.

The hunt is unfalsifiable in production: every wrong decision looks like bad
luck. So the decisions get tested against a fake SageMaker here instead.

    python3 selftest.py
"""

import os
import re
import sys
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from botocore.exceptions import ClientError  # noqa: E402

from sm_capacity_hunter import Cfg, classify, tick  # noqa: E402

PASS, FAIL, SKIP = [], [], []


def check(name, got, want):
    (PASS if got == want else FAIL).append(name)
    flag = "ok  " if got == want else "FAIL"
    print(f"  {flag} {name}" + ("" if got == want else f"\n         got {got!r}, wanted {want!r}"))


def skip(name, why):
    SKIP.append(name)
    print(f"  SKIP {name} — {why}")


def ago(seconds):
    return datetime.now(timezone.utc) - timedelta(seconds=seconds)


def missing(op):
    return ClientError({"Error": {"Code": "ValidationException",
                                  "Message": f"Could not find {op}."}}, op)


class FakeSM:
    """Just enough SageMaker to drive tick()."""

    def __init__(self, ep=None, config_type="ml.g6.2xlarge", config_count=1):
        self.ep = ep
        self.config_type = config_type
        self.config_count = config_count
        self.configs = {}
        self.models = {}
        self.log = []

    def describe_endpoint(self, EndpointName):
        if self.ep is None:
            raise missing("endpoint")
        return dict(self.ep, EndpointName=EndpointName)

    def create_endpoint(self, EndpointName, EndpointConfigName, **kw):
        self.log.append("create_endpoint")
        self.ep = {"EndpointStatus": "Creating", "EndpointConfigName": EndpointConfigName,
                   "LastModifiedTime": ago(0)}
        return {"EndpointArn": f"arn:aws:sagemaker:::endpoint/{EndpointName}"}

    def delete_endpoint(self, EndpointName):
        self.log.append("delete_endpoint")
        self.ep = None
        return {}

    def describe_endpoint_config(self, EndpointConfigName):
        if EndpointConfigName not in self.configs:
            raise missing("endpoint configuration")
        return {"EndpointConfigName": EndpointConfigName,
                "ProductionVariants": [{"InstanceType": self.config_type,
                                        "InitialInstanceCount": self.config_count}]}

    def create_endpoint_config(self, EndpointConfigName, ProductionVariants, **kw):
        self.log.append("create_endpoint_config")
        self.configs[EndpointConfigName] = ProductionVariants
        self.config_type = ProductionVariants[0]["InstanceType"]
        self.config_count = ProductionVariants[0]["InitialInstanceCount"]
        return {}

    def describe_model(self, ModelName):
        if ModelName not in self.models:
            raise missing("model")
        return {"ModelName": ModelName, "PrimaryContainer": {"Image": "img"}}

    def create_model(self, ModelName, **kw):
        self.log.append("create_model")
        self.models[ModelName] = kw
        return {}


def cfg(**over):
    base = {"region": "ap-southeast-2", "endpoint_name": "t-ep", "instance_type": "ml.g6.2xlarge",
            "count": 1, "model_name": "t-model", "image_uri": "1.dkr.ecr.ap-southeast-2.amazonaws.com/x:1",
            "role_arn": "arn:aws:iam::111111111111:role/SMExec", "sns_topic_arn": None}
    base.update(over)
    return Cfg(base)


print("\nclassify() against real SageMaker FailureReason wording")
for reason, want in [
    ("CapacityError: Unable to provision requested ML compute capacity. Please retry using a "
     "different ML instance type.", "capacity"),
    ("There is not enough capacity for instance type ml.g6.2xlarge. Please try again later.", "capacity"),
    ("Insufficient instance capacity in the requested Availability Zone.", "capacity"),
    ("The primary container for production variant AllTraffic did not pass the ping health check. "
     "Please check CloudWatch logs for this endpoint.", "fatal"),
    ("ResourceLimitExceeded: The account-level service limit 'ml.g6.2xlarge for endpoint usage' "
     "is 0 Instances.", "fatal"),
    ("Could not find model data at s3://bucket/model.tar.gz.", "fatal"),
    ("AccessDenied: User is not authorized to perform ecr:BatchGetImage", "fatal"),
    ("Something nobody has documented happened", "unknown"),
    ("", "unknown"),
]:
    check(f"classify {reason[:46]!r}", classify(reason), want)

print("\nfull acquisition cycle: absent -> create -> capacity failure -> retry -> holding")
sm = FakeSM()
c = cfg()

r = tick(sm, c)
check("absent creates the endpoint", (r["state"], r["action"]), ("hunting", "create_endpoint"))
check("model+config built on the way", sm.log[:2], ["create_model", "create_endpoint_config"])
check("config name encodes the shape", c.config_name, "t-ep-ml-g6-2xlarge-x1")

r = tick(sm, c)
check("Creating is left alone", (r["state"], r["action"]), ("hunting", "wait"))

sm.ep = {"EndpointStatus": "Failed", "EndpointConfigName": c.config_name,
         "FailureReason": "CapacityError: Unable to provision requested ML compute capacity.",
         "LastModifiedTime": ago(400)}
r = tick(sm, c)
check("capacity failure is deleted for retry", (r["state"], r["action"]), ("hunting", "delete_failed"))
check("classified as capacity", r["failure_kind"], "capacity")
check("endpoint really gone", sm.ep, None)

r = tick(sm, c)
check("next tick re-creates", r["action"], "create_endpoint")

sm.ep = {"EndpointStatus": "InService", "EndpointConfigName": c.config_name,
         "ProductionVariants": [{"CurrentInstanceCount": 1, "DesiredInstanceCount": 1}],
         "EndpointArn": "arn:...", "LastModifiedTime": ago(30)}
r = tick(sm, c)
check("InService on target = holding", (r["state"], r["action"]), ("holding", "none"))
check("held type verified from the live config", r["held_instance_type"], "ml.g6.2xlarge")
before = list(sm.log)
tick(sm, c)
check("holding writes nothing", sm.log, before)

print("\nwrong shape is reported, not silently accepted")
sm2 = FakeSM(config_type="ml.g5.2xlarge")
sm2.configs["t-ep-ml-g6-2xlarge-x1"] = []
sm2.ep = {"EndpointStatus": "InService", "EndpointConfigName": "t-ep-ml-g6-2xlarge-x1",
          "ProductionVariants": [{"CurrentInstanceCount": 1}], "LastModifiedTime": ago(9999)}
r = tick(sm2, cfg())
check("g5 while wanting g6 = holding_degraded", r["state"], "holding_degraded")
check("no destructive action taken", r["action"], "none")

sm3 = FakeSM(config_count=1)
sm3.configs["t-ep-ml-g6-2xlarge-x2"] = []
sm3.ep = {"EndpointStatus": "InService", "EndpointConfigName": "t-ep-ml-g6-2xlarge-x2",
          "ProductionVariants": [{"CurrentInstanceCount": 1}], "LastModifiedTime": ago(9999)}
r = tick(sm3, cfg(count=2))
check("1 of 2 instances = holding_degraded", r["state"], "holding_degraded")

print("\nconfiguration failures halt instead of looping forever")
for reason, kind in [
    ("The primary container for production variant AllTraffic did not pass the ping health check.", "fatal"),
    ("ResourceLimitExceeded: account-level service limit is 0 Instances.", "fatal"),
]:
    smx = FakeSM(); smx.configs[cfg().config_name] = []
    smx.ep = {"EndpointStatus": "Failed", "EndpointConfigName": cfg().config_name,
              "FailureReason": reason, "LastModifiedTime": ago(60)}
    r = tick(smx, cfg())
    check(f"halts on {kind}: {reason[:40]!r}", (r["state"], r["action"]), ("halted", "none"))
    check("nothing deleted while halted", smx.ep is not None, True)

print("\nunknown failure reasons: halt by default, retry only when told to")
mk = lambda: {"EndpointStatus": "Failed", "EndpointConfigName": cfg().config_name,
              "FailureReason": "mystery", "LastModifiedTime": ago(60)}
smu = FakeSM(); smu.ep = mk()
check("unknown halts by default", tick(smu, cfg())["state"], "halted")
smu2 = FakeSM(); smu2.ep = mk()
r = tick(smu2, cfg(retry_unknown=True))
check("--retry-unknown retries it", (r["state"], r["action"]), ("hunting", "delete_failed"))

print("\nOutOfService: grace period, then re-hunt (this is the 'hold forever' half)")
smo = FakeSM(); smo.ep = {"EndpointStatus": "OutOfService", "EndpointConfigName": cfg().config_name,
                          "LastModifiedTime": ago(60)}
r = tick(smo, cfg(reclaim_after=900))
check("inside grace = wait", (r["state"], r["action"]), ("degraded", "wait"))
smo.ep["LastModifiedTime"] = ago(1200)
r = tick(smo, cfg(reclaim_after=900))
check("past grace = delete and re-hunt", (r["state"], r["action"]), ("hunting", "delete_out_of_service"))
smo2 = FakeSM(); smo2.ep = {"EndpointStatus": "OutOfService", "EndpointConfigName": cfg().config_name,
                            "LastModifiedTime": ago(99999)}
r = tick(smo2, cfg(reclaim_after=-1))
check("reclaim disabled never deletes", (r["state"], r["action"]), ("degraded", "wait"))

print("\nread-only and dry-run modes never write")
smr = FakeSM()
tick(smr, cfg(), act=False)
check("--status on absent endpoint writes nothing", smr.log, [])
smd = FakeSM()
tick(smd, cfg(dry_run=True))
check("--dry-run writes nothing", smd.log, [])

print("\nUpdateRollbackFailed needs a human")
smh = FakeSM(); smh.ep = {"EndpointStatus": "UpdateRollbackFailed",
                          "EndpointConfigName": cfg().config_name, "LastModifiedTime": ago(60)}
check("halts", tick(smh, cfg())["state"], "halted")

# ---------------------------------------------------------------------------------
# The Lambda gets all its configuration from template.yaml. A renamed parameter or
# a typo'd env var would deploy cleanly and hunt the wrong thing, so check both
# directions of that contract here rather than in production.
# ---------------------------------------------------------------------------------
print("\ntemplate.yaml <-> code contract")

SRC = open(os.path.join(HERE, "src", "sm_capacity_hunter.py")).read()
CONSUMED = set(re.findall(r'_env(?:_int|_bool|_json|_list)?\(\s*"([A-Z_]+)"', SRC))
check("code reads the vars we expect", "REGION" in CONSUMED and "INSTANCE_TYPE" in CONSUMED, True)

try:
    import yaml
except ImportError:
    skip("template env vars are all consumed", "pyyaml not installed")
    skip("template supplies every var the Lambda needs", "pyyaml not installed")
else:
    class CfnLoader(yaml.SafeLoader):
        """CloudFormation intrinsics (!Ref, !Sub, !If) are opaque here."""

    CfnLoader.add_multi_constructor(
        "!",
        lambda loader, suffix, node: (
            loader.construct_scalar(node) if isinstance(node, yaml.ScalarNode)
            else loader.construct_sequence(node, deep=True) if isinstance(node, yaml.SequenceNode)
            else loader.construct_mapping(node, deep=True)
        ),
    )
    with open(os.path.join(HERE, "template.yaml")) as fh:
        tpl = yaml.load(fh, Loader=CfnLoader)

    fn = tpl["Resources"]["HunterFunction"]["Properties"]
    tpl_env = set(fn["Environment"]["Variables"])
    check("template env vars are all consumed", sorted(tpl_env - CONSUMED), [])

    needed = {"REGION", "INSTANCE_TYPE", "INSTANCE_COUNT", "ENDPOINT_NAME", "ROLE_ARN", "IMAGE_URI"}
    check("template supplies every var the Lambda needs", sorted(needed - tpl_env), [])
    check("handler points at this module", fn["Handler"], "sm_capacity_hunter.lambda_handler")
    check("SageMakerRoleArn has no default, so --guided must ask",
          "Default" in tpl["Parameters"]["SageMakerRoleArn"], False)
    check("ImageUri has a default, so --guided does not have to ask",
          bool(tpl["Parameters"]["ImageUri"].get("Default")), True)

    # Region is a parameter, not the stack's own region. Everything region-scoped
    # has to follow the parameter or the Lambda hunts in one region while its IAM
    # policy authorises another — which fails as AccessDenied, and AccessDenied is
    # classified fatal, so it would halt rather than mislead. Still wrong, though.
    region_default = tpl["Parameters"]["TargetRegion"]["Default"]
    check("TargetRegion defaults to ap-southeast-2", region_default, "ap-southeast-2")
    check("REGION env comes from TargetRegion, not the stack region",
          fn["Environment"]["Variables"]["REGION"], "TargetRegion")
    check("default ImageUri lives in the default TargetRegion",
          f".dkr.ecr.{region_default}." in tpl["Parameters"]["ImageUri"]["Default"], True)

    sagemaker_arns = [
        r
        for stmt in fn["Policies"][0]["Statement"]
        for r in ([stmt["Resource"]] if isinstance(stmt["Resource"], str) else stmt["Resource"])
        if ":sagemaker:" in r
    ]
    check("found the SageMaker ARNs to check", len(sagemaker_arns), 4)
    check("every SageMaker ARN is scoped to TargetRegion",
          [a for a in sagemaker_arns if "${TargetRegion}" not in a], [])
    check("no SageMaker ARN is scoped to the stack region",
          [a for a in sagemaker_arns if "${AWS::Region}" in a], [])

    # Verified against the live API: CreateEndpoint is authorised against the
    # endpoint AND the endpoint config it reads. Granting only the endpoint ARN
    # returns AccessDenied naming the endpoint-config ARN, which classify() calls
    # fatal — so the hunt would halt on its first tick, in the one code path no
    # credential-free test can reach. Guard the grant instead.
    create_ep_on = [
        r
        for stmt in fn["Policies"][0]["Statement"]
        for a in ([stmt["Action"]] if isinstance(stmt["Action"], str) else stmt["Action"])
        if a == "sagemaker:CreateEndpoint"
        for r in ([stmt["Resource"]] if isinstance(stmt["Resource"], str) else stmt["Resource"])
    ]
    check("CreateEndpoint is granted on the endpoint",
          any(":endpoint/" in r for r in create_ep_on), True)
    check("CreateEndpoint is also granted on the endpoint config it reads",
          any(":endpoint-config/" in r for r in create_ep_on), True)
    check("the release command targets the endpoint's region",
          "${TargetRegion}" in tpl["Outputs"]["ReleaseCapacity"]["Value"], True)

    # The default container must be self-sufficient: no model.tar.gz to supply.
    check("default ModelDataUrl is blank", tpl["Parameters"]["ModelDataUrl"]["Default"], "")
    import json as _json
    env_default = _json.loads(tpl["Parameters"]["ContainerEnvJson"]["Default"])
    check("default container env is valid JSON with a model id",
          "HF_MODEL_ID" in env_default, True)

print("\nmalformed CONTAINER_ENV fails loudly rather than silently emptying")
saved = os.environ.get("CONTAINER_ENV")
os.environ["CONTAINER_ENV"] = "{not json"
try:
    Cfg({"endpoint_name": "t-ep"})
    check("bad CONTAINER_ENV raises", False, True)
except SystemExit as e:
    check("bad CONTAINER_ENV raises", "not valid JSON" in str(e), True)
finally:
    if saved is None:
        os.environ.pop("CONTAINER_ENV", None)
    else:
        os.environ["CONTAINER_ENV"] = saved

print("\nenvironment-only configuration, exactly as the Lambda receives it")
env = {"REGION": "ap-southeast-2", "INSTANCE_TYPE": "ml.g6.4xlarge", "INSTANCE_COUNT": "3",
       "ENDPOINT_NAME": "prod-hold", "MODEL_NAME": "prod-hold-model",
       "ROLE_ARN": "arn:aws:iam::222222222222:role/SM", "IMAGE_URI": "x/y:1",
       "CONTAINER_ENV": '{"HF_MODEL_ID":"distilbert-base-uncased"}',
       "MODEL_DATA_URL": "", "RETRY_UNKNOWN": "true", "RECLAIM_AFTER_SECONDS": "-1"}
saved_env = {k: os.environ.get(k) for k in env}
os.environ.update(env)
try:
    ec = Cfg()
    check("instance type from env", ec.instance_type, "ml.g6.4xlarge")
    check("count coerced to int", ec.count, 3)
    check("retry_unknown coerced to bool", ec.retry_unknown, True)
    check("reclaim -1 becomes None (never delete)", ec.reclaim_after, None)
    check("blank MODEL_DATA_URL becomes None, not ''", ec.model_data_url, None)
    check("container env parsed", ec.container_env, {"HF_MODEL_ID": "distilbert-base-uncased"})
    check("config name from env", ec.config_name, "prod-hold-ml-g6-4xlarge-x3")
finally:
    for k, v in saved_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

print(f"\n{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
if FAIL:
    print("failures: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
