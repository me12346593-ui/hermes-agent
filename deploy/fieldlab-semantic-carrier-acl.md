# FieldLab semantic carrier socket ACL candidate

Status: OFFLINE CANDIDATE ONLY. No server, systemd, ACL, credential, or production mutation is authorized by this file.

## Boundary

- Operation: `fieldlab.semantic.execute.v1`
- Socket: `/run/hermes-fieldlab/semantic.sock`
- Hermes gateway principal: existing `ubuntu`
- FieldLab runtime principal: existing `fieldlab`
- New service: NO
- New credential: NO
- Existing Linux user/group membership changes: NO
- Supply socket/plugin/config changes: NO
- FieldLab access to `/home/ubuntu/.hermes`: DENY / unchanged

The dedicated parent directory is deployment-owned. Application code must not create it, chown it, or grant ACLs.

## Proposed POSIX ACL model

Parent directory access ACL:

```text
/run/hermes-fieldlab
owner = ubuntu
user::rwx
user:fieldlab:--x
group::---
mask::--x
other::---
```

The named `fieldlab` entry permits traversal to the one known socket path but not directory listing.

Default ACL for new socket nodes created in this dedicated directory:

```text
default:user::rw-
default:user:fieldlab:rw-
default:group::---
default:mask::rw-
default:other::---
```

Expected socket effective access after Hermes binds `semantic.sock`:

```text
owner ubuntu: rw
user fieldlab: rw
group: ---
other: ---
```

The deployment gate must verify the actual access ACL on the socket after creation/restart. A PASS requires:

- `fieldlab` can connect to `/run/hermes-fieldlab/semantic.sock`;
- ordinary local principals cannot connect;
- `fieldlab` still cannot read `/home/ubuntu/.hermes` or credential material;
- Hermes receives no permission to `/srv/fieldlab/data/runtime.sqlite` or any FieldLab write surface;
- Supply socket/plugin/config remain byte-for-byte / identity unchanged as applicable.

## Rollback boundary

Rollback removes only:

1. the FieldLab semantic socket surface;
2. the dedicated parent ACL/directory materialization;
3. the FieldLab I3 caller binding / semantic operation code through the normal code rollback path.

Rollback does not touch FieldLab DB/business data, Protocol V2.7, RuleSet, CalculationKernel, StateOwner, Hermes credentials, Supply code/config/socket, or Linux user/group membership.

No live commands are authorized by this candidate.
