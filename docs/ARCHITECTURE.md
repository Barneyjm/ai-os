# meetkatOS Architecture

## Layer Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  AI Shell   │  │   Voice I/O │  │   Web/Remote API    │  │
│  │   (TUI)     │  │  (optional) │  │     (optional)      │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
└─────────┼────────────────┼───────────────────┼──────────────┘
          │                │                   │
          └────────────────┼───────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    System Agent                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Tool Registry                                       │    │
│  │  - filesystem ops    - process management            │    │
│  │  - package manager   - network configuration         │    │
│  │  - service control   - hardware interfaces           │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Agency Policy Engine                                │    │
│  │  - per-domain policies  - profile switching          │    │
│  │  - adaptive learning    - consent management         │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Inference Runtime                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Model Loader │  │  KV Cache    │  │  Batch Scheduler │   │
│  │ & Hot-swap   │  │  Manager     │  │  (multi-client)  │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
│  llama.cpp server @ /run/ai-runtime.sock                     │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Linux Kernel                               │
└─────────────────────────────────────────────────────────────┘
```

## Agency Policy System

The policy system controls AI autonomy across domains:

```
Agency Levels:
  observe    → Watch and log, never act
  suggest    → Notify user of possible actions
  confirm    → Ask permission before acting
  auto       → Act automatically, notify after
  autonomous → Act silently
```

Policies are resolved in order:
1. Sensitive path overrides (e.g., ~/.ssh always confirms)
2. Specific operation (e.g., domains.filesystem.write_user_files)
3. Domain default (e.g., domains.filesystem.level)
4. Global default (defaults.level)

## Profiles

Quick presets for different contexts:
- **default**: Balanced for daily use
- **cautious**: Maximum confirmation
- **focus**: Minimize interruptions
- **away**: Full autonomy
- **gaming**: Minimal AI activity
- **presentation**: No interruptions

## Security Model

- Per-domain capability controls
- Sensitive path detection
- Protected process list
- Response tracking for adaptive learning
- Audit trail of AI actions
