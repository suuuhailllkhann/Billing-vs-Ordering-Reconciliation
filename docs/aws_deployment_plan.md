# Future AWS deployment plan

## Status and boundaries

This document is a planning artifact only. AWS deployment has **not** been performed,
and no AWS resources have been created. The currently validated deployment is local
Docker Compose with FastAPI and PostgreSQL. The design below is a possible future path;
it is not a production-readiness claim, an authorization to deploy, or a commitment to
ongoing AWS costs.

All current development and validation use synthetic data. Real pharmacy, patient, PHI,
PII, or company data must not be introduced as part of any future deployment without a
separate security, privacy, compliance, and operational review.

## Proposed initial architecture

```text
Amazon ECR
    ↓
EC2
    ↓
Docker Compose
├── FastAPI
└── PostgreSQL
      ↓
EBS persistent storage

Application/container logs
    ↓
CloudWatch

Possible later evolution:
PostgreSQL → RDS
MLflow artifacts → S3
ECS/Fargate → optional future upgrade
```

The initial proposal uses one EC2 host running the already-tested Compose services. The
FastAPI image would be built through an approved release process and stored in Amazon
ECR. EC2 would pull that immutable image and run FastAPI alongside PostgreSQL using the
existing service topology. PostgreSQL data would reside on an attached EBS volume rather
than ephemeral instance storage. Application and container logs would be forwarded to
CloudWatch without changing the safe logging policy or including request payloads.

EC2 plus Docker Compose is proposed first because it is simple, has relatively low
operational complexity, and closely matches the locally validated architecture. This
reduces the number of architectural changes in an initial deployment exercise. It does
not provide the managed availability, database isolation, automated scaling, or reduced
host-management burden that a more mature architecture may require.

## Future deployment responsibilities

A future implementation would need a reviewed process for:

- publishing versioned application images to ECR;
- provisioning and patching the EC2 host;
- attaching, formatting, mounting, encrypting, and monitoring EBS storage;
- supplying runtime configuration and secrets without embedding them in images;
- applying database migrations explicitly and safely;
- starting and validating the Compose services;
- forwarding privacy-safe application/container logs to CloudWatch; and
- defining rollback, recovery, incident-response, and resource-retirement procedures.

The locked MLflow model and its required artifacts would also need a controlled,
read-only delivery mechanism. This plan does not choose or implement that mechanism.
Raw datasets, patient-level predictions, and identifiers must not be placed in images,
logs, or general-purpose artifact storage.

## Security and operational considerations

Before any AWS deployment, the design must be reviewed and expanded to include:

- IAM roles and policies following least privilege, with no long-lived credentials baked
  into images or source control;
- secrets and configuration supplied through appropriate AWS-managed mechanisms rather
  than Compose files, shell history, image layers, or committed `.env` files;
- narrowly restricted security groups, with PostgreSQL not exposed publicly;
- HTTPS/TLS for external traffic and encryption requirements for data in transit and at
  rest;
- EBS snapshots, tested backup/restore procedures, retention rules, and recovery targets;
- CloudWatch monitoring, alerting, log retention, and continued exclusion of sensitive
  request or patient data;
- database durability, capacity, patching, integrity, and failure-recovery procedures;
- operating-system and container vulnerability management;
- access auditing and incident-response ownership; and
- an explicit cost estimate, budget, and resource lifecycle policy before resources are
  created.

These controls are considerations only; none are implemented by Phase 4.

## Possible later evolution

If operational requirements justify additional complexity, PostgreSQL could move to
Amazon RDS for a more managed database boundary, MLflow artifacts could move to a
properly restricted and encrypted S3 location, and service orchestration could move from
EC2 Compose to ECS/Fargate. Each change would require separate design, privacy, security,
cost, migration, and validation work. These are optional future directions, not current
project capabilities or approved deployment decisions.

## Readiness statement

This plan documents direction only. It does not demonstrate AWS availability, security,
scalability, compliance, backup recovery, cost suitability, or production performance.
The local Docker Compose validation remains the only deployment validation completed to
date.
