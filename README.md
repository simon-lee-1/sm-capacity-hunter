> [!IMPORTANT]
> **This repository has moved to [simon-lee-1/sagemaker-inference-endpoint-capacity](https://github.com/simon-lee-1/sagemaker-inference-endpoint-capacity).**
>
> It is archived and no longer maintained. The project was renamed and the
> module is now `src/capacity.py`. Everything here is superseded; nothing was
> lost in the move.

# sm-capacity-hunter

Retries a SageMaker real-time endpoint until scarce on-demand GPU capacity lands,
then holds it and keeps watch. Region and instance type are parameters;
`ap-southeast-2` and `ml.g6.2xlarge` are the defaults.

```
template.yaml                 SAM app: parameters, Lambda, schedule, scoped IAM
src/sm_capacity_hunter.py     the state machine (CLI + lambda_handler in one file)
src/Makefile                  build step: no dependencies, just copies the module
selftest.py                   60 credential-free checks
```

## Deploy

```bash
sam build
sam deploy --guided
```

`--guided` prompts for each parameter with its default and saves your answers to
`samconfig.toml`; later deploys are just `sam deploy`.

**`SageMakerRoleArn` is the one value you must supply.** It has no default
because a wrong guess fails minutes into the first attempt looking like scarcity.
If you don't have one:

```bash
aws iam create-role --role-name SageMakerCapacityHoldRole \
  --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow",
    "Principal":{"Service":"sagemaker.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam attach-role-policy --role-name SageMakerCapacityHoldRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
```

| Parameter | Default | Notes |
|---|---|---|
| `SageMakerRoleArn` | **none — you fill this in** | validated as an IAM role ARN |
| `TargetRegion` | `ap-southeast-2` | where the endpoint goes; see [two regions](#two-regions) |
| `InstanceType` | `ml.g6.2xlarge` | any `ml.*` inference type |
| `InstanceCount` | `1` | your quota must be at least this |
| `EndpointName` | `capacity-hold-ep` | lowercase, so the scoped IAM policy stays correct |
| `ImageUri` | HuggingFace inference DLC | must live in `TargetRegion`; see below |
| `ContainerEnvJson` | `{"HF_MODEL_ID":"distilbert-base-uncased","HF_TASK":"fill-mask"}` | why no `model.tar.gz` is needed |
| `ModelDataUrl` | *(blank)* | set if your image needs an S3 artefact |
| `Schedule` | `rate(1 minute)` | one tick = one step |
| `RetryUnknown` | `false` | see [classification](#classification) |
| `ReclaimAfterSeconds` | `900` | OutOfService grace before re-hunting |
| `NotifyEmail` | *(blank)* | creates an SNS topic and subscribes you |

Changing any of them needs no code change:

```bash
sam deploy --parameter-overrides InstanceType=ml.g6.4xlarge TargetRegion=us-west-2 ImageUri=...
```

Quote values containing spaces or SAM splits them into separate parameters and
the deploy rolls back — `Schedule` is the one that bites:
`'Schedule="rate(5 minutes)"'`.

Don't hand-edit the Lambda's environment variables in the console; the next
`sam deploy` silently reverts them. The code also reads `SUBNET_IDS` and
`SECURITY_GROUP_IDS` (both required together, for a `VpcConfig`), `DRY_RUN`,
`NOTIFY_WINDOW_SECONDS`, `TAGS`, and for `--loop` only `HUNT_POLL_SECONDS` /
`HOLD_POLL_SECONDS`. The template exposes none of these; set them on the
function or pass the matching CLI flag.

### Two regions

`TargetRegion` is where SageMaker creates the endpoint; the Lambda passes it to
every SageMaker call and the IAM policy is scoped to it. The **stack region**,
chosen by `sam deploy`, only holds the Lambda, its log group and the EventBridge
rule. Usually you want them equal, but differing is supported, so one hunter can
chase capacity in a region you don't otherwise deploy into.

The trap is forgetting, because then the endpoint won't appear in the console
region you deployed to. So `--preflight` prints both and flags the divergence,
every tick logs the `region` it is hunting, and the stack exports a
`TargetRegion` output. `ImageUri` must live in `TargetRegion` too — preflight
calls a mismatch `FAIL` rather than letting every attempt die looking like
scarcity.

### The default `ImageUri`

The HuggingFace inference DLC, paired with `HF_MODEL_ID` so it downloads a small
model at startup and needs **no S3 artefact** — a role ARN is all you need to
start hunting. Confirmed present in `ap-southeast-2` on 2026-09-15 (digest
`sha256:c1fa2c7f…`), but DLC tags get retired; a stale one fails as an image-pull
error, which `classify()` calls fatal, so the hunt halts rather than loops.

Any image works if it answers `GET /ping` with 200 and accepts
`POST /invocations`. `pytorch-inference` needs a `ModelDataUrl`; the HuggingFace
one with `HF_MODEL_ID` doesn't. To resolve a current tag:

```bash
python3 -c "import sagemaker; print(sagemaker.image_uris.retrieve(
  'pytorch','ap-southeast-2',version='2.5.1',
  instance_type='ml.g6.2xlarge',image_scope='inference'))"
```

## One tick

| Endpoint state | Action | Reported as |
|---|---|---|
| absent | create model + config, `CreateEndpoint` | `hunting` |
| `Creating` / `Deleting` / `Updating` | nothing | `hunting` |
| `Failed`, capacity reason | `DeleteEndpoint` so the next tick retries | `hunting` |
| `Failed`, config reason | stop, notify, leave it `Failed` | `halted` |
| `Failed`, unrecognised reason | stop unless `RetryUnknown=true` | `halted` |
| `OutOfService` | wait `ReclaimAfterSeconds`, then re-hunt | `degraded` |
| `InService`, right type and count | nothing | `holding` |
| `InService`, wrong type or short | nothing, but say so | `holding_degraded` |

### Classification

A container failing `/ping`, a missing `model.tar.gz` and a zero quota all produce
a `Failed` endpoint, indistinguishable from scarcity if you only count attempts —
retry those forever and the tool looks patient but is dead. So `FailureReason` is
classified: recognised capacity errors get deleted and retried, anything else
halts and notifies. `RetryUnknown=true` overrides that once you've read the reason
and know it's transient. Preflight also checks `<type> for endpoint usage`, since
a quota of 0 can never succeed, reporting `UNKNOWN` — never `OK` — if Service
Quotas can't be read.

## Why Lambda

Lambda's 15-minute ceiling doesn't bite because **the loop isn't inside the
process**. Each invocation advances the state machine one step, using the live
endpoint as its only state store — no DynamoDB, nothing to resume, and two
overlapping ticks cost one wasted API call at worst. That also gives you the
watchdog for free: the tick that hunts is the tick that spots `OutOfService`.

## Operating it

These address the Lambda, so they take the **stack** region:

```bash
aws logs tail /aws/lambda/<stack-name> --follow --region <stack-region>

# checks, inside the deployed function
aws lambda invoke --function-name <stack-name> --region <stack-region> \
  --payload '{"preflight":true}' --cli-binary-format raw-in-base64-out /dev/stdout

# pause without deleting anything
aws events disable-rule --name <stack-name>-tick --region <stack-region>
```

Anything addressing the endpoint takes `TargetRegion` instead. The stack outputs
print all of them with the right region already substituted.

The same module runs locally against any credentials: `--preflight`, `--status`
and `--dry-run` write nothing, `--once` advances one step, `--loop` hunts in the
foreground, `--release` deletes the endpoint.

```bash
python3 src/sm_capacity_hunter.py --status --endpoint-name capacity-hold-ep
```

## Once it lands

`"state":"holding"` means the instances are yours for as long as the endpoint
exists. SageMaker doesn't reclaim on-demand endpoint instances the way Spot does,
and there's no ODCR path for `ml.` instances, which is why polling is the only
lever. To keep it: leave the schedule enabled (it's the watchdog), don't attach
autoscaling that can scale to zero, and deny `sagemaker:DeleteEndpoint` on this
endpoint ARN to anyone who doesn't need it — handing capacity back is a one-way
door.

**`sam delete` does not delete the endpoint.** The Lambda creates it at runtime,
not CloudFormation, so it outlives the stack and keeps billing:

```bash
python3 src/sm_capacity_hunter.py --release --yes --region <TargetRegion> \
  --endpoint-name capacity-hold-ep
```

## License

MIT — see [LICENSE](LICENSE).
