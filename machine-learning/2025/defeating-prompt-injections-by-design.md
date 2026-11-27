# Meta Information

- URL: [Defeating Prompt Injections by Design](https://arxiv.org/abs/2503.18813)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html) / [Creative Commons BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Reference: Debenedetti, E., Shumailov, I., Fan, T., Hayes, J., Carlini, N., Fabian, D., Kern, C., Shi, C., Terzis, A., & Tramèr, F. (2025). Defeating Prompt Injections by Design. arXiv:2503.18813.

# Defeating Prompt Injections by Design (CaMeL)

## Background: Prompt Injection Attacks

LLM agents retrieve data from external sources (emails, documents, search results) and execute tool calls based on that data. **Prompt injection attacks** exploit this by embedding malicious instructions in untrusted data, causing the agent to execute attacker-controlled actions (e.g., exfiltrating private data, sending unauthorized emails, modifying files).

Existing defenses—such as input filtering, fine-tuning for robustness, or system prompt hardening—all rely on the LLM to distinguish trusted from untrusted content. Since the LLM cannot reliably make this distinction (it sees all text the same way), these defenses are fundamentally brittle.

> [!IMPORTANT]
> CaMeL's key insight: the defense must be implemented **outside** the LLM, in a protective layer that does not rely on the LLM's judgment. This is analogous to how modern operating systems use hardware memory protection rather than trusting programs to respect each other's memory.

## CaMeL Architecture

CaMeL (Capability-based Machine Learning) places a **security layer** between the LLM and the execution environment. The architecture has two main components:

1. **A Python interpreter** that executes a restricted subset of Python, tracking data provenance through a taint analysis system.
2. **A security policy engine** that enforces capability-based access control on all tool calls.

### Two-LLM Design

CaMeL uses two LLM calls in sequence:

1. **Planner LLM (trusted)**: Receives only the trusted user query and converts it into a Python program. This program specifies the control flow—which tools to call, in what order, with what arguments. The Planner only sees trusted input, so it cannot be injected.
2. **Reader LLM (untrusted)**: Reads and summarizes untrusted external data (e.g., email contents, search results). Its output is marked as **untrusted** and wrapped in `CaMeLValue` objects with appropriate capability metadata.

The key guarantee: **untrusted data returned by the Reader can never alter the Python program's control flow**, because the program is already fixed by the Planner before any external data is retrieved.

```
User Query (trusted)
    │
    ▼
[Planner LLM] ──► Python program (control flow)
                          │
                          ▼
                  [CaMeL Interpreter]
                     │        │
              Tool calls    Reads untrusted data
                     │        │
              [Policy Engine] [Reader LLM]
                     │        │
              Allowed/Denied  CaMeLValue (tainted)
```

### CaMeLValue: Taint Tracking

Every value in the CaMeL interpreter is wrapped in a `CaMeLValue` object with three fields:

- `_python_value`: The underlying Python object (string, int, list, etc.)
- `_metadata`: A `Capabilities` object encoding the value's security classification (e.g., `Tool("email_client")`, `Public`)
- `_dependencies`: A tuple of upstream `CaMeLValue` objects forming a data provenance graph

When values are combined (e.g., string concatenation, list membership), the resulting value inherits the union of all its inputs' dependencies and capabilities. This is **taint propagation**: any operation on a tainted value produces a tainted result.

**Example**: If `email_body` is tainted with capability `Tool("get_email")`, then `subject = email_body.split(":")[0]` produces a `subject` value also tainted with `Tool("get_email")`. Attempting to use `subject` as an argument to `send_email(to=subject)` will trigger a policy check.

```python
# Pseudocode for taint propagation in binary operations
def eval_binary_op(left: CaMeLValue, right: CaMeLValue, op) -> CaMeLValue:
    result_value = op(left._python_value, right._python_value)
    merged_deps = left._dependencies + right._dependencies
    merged_caps = Capabilities.merge(left._metadata, right._metadata)
    return CaMeLValue(result_value, merged_caps, merged_deps)
```

### Capability-Based Access Control

Before any tool call executes, the **security policy engine** validates it. The engine checks:

1. **Side-effect exemption**: Read-only tools (e.g., `query_ai_assistant`) are always allowed.
2. **Dependency analysis**: If the tool call's arguments contain non-public tainted values, and the tool has side effects (e.g., sending an email, writing a file), the call is **denied**.
3. **Policy matching**: Named policies can explicitly allow specific data flows. For example, a policy can permit sending the contents of a specific trusted document to a specific email address.

The default behavior is **deny-by-default**: if no policy matches, the engine returns `Denied("No security policy matched for tool. Defaulting to denial.")`.

```python
# Pseudocode for policy check
def check_policy(tool_name, kwargs) -> SecurityPolicyResult:
    if tool_name in no_side_effect_tools:
        return Allowed()
    if any_kwarg_has_private_data(kwargs):
        return Denied("Data is not public.")
    for policy in policies:
        if fnmatch(tool_name, policy.pattern):
            return policy.check(kwargs)
    return Denied("No security policy matched. Defaulting to denial.")
```

### Control Flow Isolation

One subtle but critical property: **control flow decisions must not be influenced by untrusted data**. In the CaMeL interpreter, the `if`-statement evaluator explicitly removes the condition's taint from the body's dependency set:

```python
def _eval_if(test: CaMeLValue, body, orelse):
    # After using test to decide which branch to execute,
    # remove test from the dependency set of results
    dependencies = list(dependencies)
    dependencies.remove(test)
    # Execute the appropriate branch
```

This prevents an attacker from encoding information into a conditional outcome (e.g., "if the secret key starts with 'A', call tool X, else call tool Y").

## Security Guarantees

CaMeL provides **provable security** against prompt injection in a formal sense: if the Planner correctly generates a policy-compliant Python program for the user's intent, and the policy engine correctly enforces policies, then untrusted data retrieved during execution cannot cause unauthorized tool calls or data exfiltration.

The guarantee holds **regardless of whether the underlying LLM is vulnerable to prompt injection**. The security does not rely on the LLM's ability to detect malicious instructions.

> [!NOTE]
> "CaMeL creates a protective system layer around the LLM, securing it even when underlying models are susceptible to attacks." — Abstract

## Comparison with Related Defenses

| Defense | Requires LLM robustness | Modifies LLM | Provable security |
|---|---|---|---|
| Input/output filtering | Yes (relies on classifier) | No | No |
| Fine-tuning (e.g., SecAlign) | Reduces reliance | Yes | No |
| Prompt delimiters / spotlighting | Yes (LLM must respect) | No | No |
| **CaMeL** | **No** | **No** | **Yes (for covered attacks)** |

**vs. StruQ / SecAlign**: Fine-tuning-based defenses (StruQ, SecAlign) train the LLM to ignore injected instructions. They reduce attack success rates but do not eliminate them and require model modifications. CaMeL requires no model changes.

**vs. Prompt hardening / spotlighting**: These approaches instruct the LLM to treat certain text as untrusted. They rely on LLM compliance, which can be overridden by strong injection attacks.

**vs. Classical taint analysis**: CaMeL adapts information flow control from systems security (taint tracking, capability systems, label-based access control) to the LLM agent setting.

## Limitations

- **Planner correctness**: If the Planner LLM generates an incorrect or insecure program (e.g., passing untrusted data through a policy that shouldn't apply), the security guarantee weakens.
- **Policy specification burden**: Users must define security policies specifying which data flows are authorized. Incorrect or overly permissive policies undermine security.
- **Restricted Python subset**: The interpreter supports only a subset of Python, which may limit the complexity of agent tasks.
- **Utility-security tradeoff**: Strict enforcement may block legitimate operations, reducing task completion rates.

# Experiments

- **Benchmark**: AgentDojo (NeurIPS 2024) — a benchmark for evaluating prompt injection attacks and defenses for LLM agents
- **Task**: 97 user tasks across multiple task suites (workspace, banking, travel, etc.) with corresponding injection attacks
- **Metric**: Fraction of tasks completed correctly under attack, with and without provable security guarantees
- **Results**:
  - CaMeL achieves **77% of tasks completed** with provable security guarantees
  - Without any defense, the same model completes **84% of tasks** (utility upper bound)
  - CaMeL's utility cost (~7%) is substantially lower than most defenses
  - Baseline defenses (prompt hardening, input filtering) either leave significant vulnerability or impose high utility costs
- **Models tested**: Multiple underlying LLMs; CaMeL's security guarantee is model-agnostic
- **Code**: [google-research/camel-prompt-injection](https://github.com/google-research/camel-prompt-injection)

> [!TIP]
> AgentDojo benchmark: [agentdojo.spylab.ai](https://agentdojo.spylab.ai/) — developed at ETH Zurich / Invariant Labs, used by US and UK AI Safety Institutes to demonstrate LLM agent vulnerability.

> [!CAUTION]
> The 77% figure refers to tasks solved with **provable** security guarantees. The actual attack success rate against CaMeL may be even lower for weaker attacks. The paper's code repository notes the interpreter "likely contains bugs" and is a research prototype, not production-ready.
