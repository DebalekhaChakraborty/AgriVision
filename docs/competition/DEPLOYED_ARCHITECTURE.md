# AgriVision deployed architecture

Every box below is a service that is actually used. Nothing is drawn because it
would look impressive: API Gateway, Lambda, Step Functions, SQS, EventBridge,
SageMaker, Cognito and a VPC were all considered and none had a requirement this
service could point at.

```mermaid
flowchart TD
    J([Judge / client<br/>one fruit per capture])

    J -->|"HTTPS POST /inspect<br/>managed TLS certificate"| AR

    subgraph AWS["AWS · us-east-1 · account 341181499761"]
        AR["**App Runner**<br/>agrivision-inspection<br/>1 vCPU · 2 GB · 1 instance<br/>health check → /health"]

        subgraph C["Container · non-root uid 10001 · OpenCV 5.0.0"]
            API["**FastAPI**<br/>validate upload by content<br/>not by filename"]
            ORCH["**Agentic Vision orchestrator**<br/>18-state machine<br/>15-tool closed registry<br/>1 remediation · 1 inference"]
            CV["**OpenCV 5 perception**<br/>foreground · quality · artefacts"]
            DNN["**cv2.dnn**<br/>MobileNetV3 · visible condition"]
        end

        AR --> API --> ORCH --> CV
        CV -->|"evidence"| ORCH
        ORCH -->|"only from<br/>ELIGIBLE_FOR_INFERENCE"| DNN
        DNN -->|"condition + confidence<br/>recorded, not thresholded"| ORCH

        S3[("**S3**<br/>model artifact<br/>private · encrypted · versioned")]
        DDB[("**DynamoDB**<br/>run traces<br/>on-demand · 14-day TTL")]
        CW["**CloudWatch Logs**<br/>structured JSON per step"]
        ECR[("**ECR**<br/>agrivision:phase4<br/>scan on push")]

        S3 -.->|"startup fetch<br/>→ SHA-256 verify<br/>→ fail closed"| C
        ORCH -->|"trace + decisions<br/>content hashes only"| DDB
        API -->|"run_id · state · tool · action<br/>reason_code · duration_ms"| CW
        ECR -.->|"image pull"| AR

        IAM["**IAM instance role**<br/>s3:GetObject on models/*<br/>dynamodb:PutItem/GetItem<br/>nothing else"]
        IAM -.-> C
    end

    ORCH -->|"200 + structured result<br/>+ trace_url"| J
    DDB -->|"GET /inspection/{run_id}/trace"| J

    style AR fill:#e8f0fe,stroke:#1a73e8
    style CV fill:#e6f4ea,stroke:#137333
    style DNN fill:#e6f4ea,stroke:#137333
    style ORCH fill:#fef7e0,stroke:#b06000
    style IAM fill:#fce8e6,stroke:#c5221f
```

## Request path

```mermaid
sequenceDiagram
    participant J as Judge
    participant A as FastAPI
    participant O as Orchestrator
    participant D as DynamoDB
    participant L as CloudWatch

    J->>A: POST /inspect (multipart JPEG)
    A->>A: sniff magic bytes, decode, bound size and dimensions
    alt not a usable image
        A-->>J: 400 / 413 / 415 with a reason code
    else usable
        A->>O: run_inspection(decoded array)
        O->>O: segment → measure → decide (OpenCV 5)
        opt capture acceptable
            O->>O: run condition model
        end
        O->>L: one JSON line per agent step
        O->>D: trace + decisions (content hashes only)
        Note over D: a write failure is counted and<br/>logged; the result is unchanged
        O-->>A: terminal state + trace
        A-->>J: 200 + result (even for REQUEST_RECAPTURE)
    end
    J->>D: GET /inspection/{run_id}/trace
    D-->>J: full causal trace
```

## Status semantics

A refused capture is a **successful inspection**. The system did what it was
built to do; the photograph was the problem. Answering `500` would tell a client
AgriVision is broken.

| Status | Meaning |
| --- | --- |
| `200` | The loop reached a terminal state — including `REQUEST_RECAPTURE` and `REQUEST_HUMAN_REVIEW` |
| `400` | Not a usable image |
| `413` | Upload above the size limit |
| `415` | Not a supported image format |
| `503` | Not ready — no verified model artifact — or the trace store is unreachable |
| `500` | Internal fault, with a safe body and no internals |

Clients distinguish "inspected" from "needs another photograph" by
`requested_human_action`, never by the HTTP status.

Full method and evidence:
[`PHASE4_AWS_DEPLOYMENT.md`](PHASE4_AWS_DEPLOYMENT.md).
Agent loop itself: [`AGENT_WORKFLOW.md`](AGENT_WORKFLOW.md).
