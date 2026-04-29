# Meta Information

- URL: [[1205.0820] DNS-based Ingress Load Balancing: An Experimental Evaluation](https://arxiv.org/abs/1205.0820)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Kanuparthy, P., Matthews, W., & Dovrolis, C. (2012). DNS-based Ingress Load Balancing: An Experimental Evaluation. arXiv preprint arXiv:1205.0820.

# Problem Setting

**Ingress Traffic Engineering (ITE)** controls which incoming WAN link a multihomed data center uses for traffic from external clients. Multihomed networks announce the same prefix over multiple upstream providers, giving them a choice of ingress path. The goal is to balance utilization across those links to avoid congestion and maximize throughput.

Two families of ITE mechanisms exist:

| Approach | Mechanism | Granularity |
|---|---|---|
| BGP-based | Selective prefix de-aggregation or AS-path prepending | Per-prefix (coarse) |
| DNS-based | Authoritative DNS returns different IP addresses per link | Per-DNS-request (fine) |

This paper focuses on **DNS-based ITE** because it operates at a finer granularity without requiring BGP policy changes. The authoritative DNS server for a service returns one of several IP addresses (one per ingress link), effectively steering each client cohort to a different link.

## System Architecture

The prototype deployed at Georgia Tech's GTLIB software mirror consists of three components:

1. **Authoritative DNS server** — receives DNS A record queries from LDNS (recursive resolvers), selects an IP address according to the load balancing algorithm, and responds with the chosen address and a configured TTL.
2. **Monitoring process** — measures interface utilization continuously using a sliding window of width $W = n \times w$ seconds (with $w = 100$ ms per slot).
3. **Web server** — Apache on Linux serving file download requests from clients.

The system has **two ingress links** in the evaluation, so each DNS response directs client traffic to one of the two interfaces.

## Key Challenge: Decision-to-Traffic Delay

DNS-based ITE suffers from a fundamental timing mismatch. When the DNS server assigns a client to link $i$, actual traffic arrives with delay $\delta$ (the time for the TCP connection to be established and data to start flowing). During this interval, the DNS server may assign additional clients, leading to **over-subscription** of one link before any corrective measurement feedback arrives. If $\delta > 1/\lambda$ (where $\lambda$ is the DNS request arrival rate), multiple back-to-back assignments land on the same link before the first connection's load is visible.

# Workload Characterization

## TTL Compliance

Analysis of a 24-hour GTLIB trace (April 10, 2008) with 46,400 DNS A record resolutions from 2,864 unique LDNS servers shows:

- **60% of LDNS servers** fully respected the advertised 8-hour TTL, reissuing DNS queries only after expiry.
- **40% violated** the advertised TTL, re-querying earlier (either much earlier or ignoring TTL entirely).

TTL violations reduce effective DNS request granularity: a violating LDNS re-queries frequently, allowing the ITE system to reassign that LDNS's clients more often. Compliant LDNS servers cache the response for the full TTL duration, locking in their assignment.

## Client and Transfer Distributions

- **Client-to-LDNS association**: 92% of clients could be attributed to a unique LDNS server, validating the assumption that an LDNS's clients can be approximated as a fixed cohort.
- **Clients per LDNS**: Follows a **Pareto (heavy-tailed) distribution** — most LDNS servers represent a small number of clients, but a few represent very large client pools. Large LDNS cohorts make load balancing coarser because a single DNS response commits a large traffic burst to one link.
- **Transfer sizes**: Follow a **lognormal distribution**, spanning several orders of magnitude. Large transfers contribute disproportionate load per DNS decision and increase variability.

# Load Balancing Algorithms

## Round-Robin (RR)

RR cycles through the available IP addresses in fixed order, irrespective of current link utilization. It achieves statistical fairness only if transfer sizes are small and roughly equal, so that each assignment contributes approximately the same load. Under heavy-tailed transfer size distributions or large per-LDNS client counts, RR produces significant imbalance.

## Measurement-Based (MB)

MB selects the IP address corresponding to the ingress link with **minimum measured utilization** over the most recent measurement window $W$:

```
Algorithm: MB Load Balancing
Input:  DNS A record query q
        U[1..L]: measured utilization for each link l over window W
Output: IP address a* for the chosen link

1. For each link l in {1, ..., L}:
     U[l] ← (bytes received on link l in last W seconds) / (link capacity × W)
2. l* ← argmin_{l} U[l]
3. Return IP address a* corresponding to l*
```

The measurement window $W = n \times w$ seconds is the key hyperparameter. Too short a window ($W \approx 0.1$ s) produces noisy utilization estimates with high variance, causing the algorithm to oscillate. Too long a window ($W \approx 30$ s) makes utilization estimates stale, especially under bursty workloads. The paper finds that $W \in [5, 15]$ seconds gives the best accuracy for TCP-based file download workloads.

# Mathematical Model

## DNS Request Rate

Let $n$ be the number of clients behind an LDNS server and $r$ the per-client request arrival rate. The effective DNS request rate $\lambda$ that the ITE system observes depends on whether the LDNS caches responses:

```math
\begin{align}
  \lambda &= n r \quad \text{(non-caching LDNS, TTL ignored)} \\
  \lambda &= \min\!\left\{n r,\; \frac{1}{T}\right\} \quad \text{(caching LDNS with TTL } T \text{)}
\end{align}
```

For a compliant caching LDNS with TTL $T$, even if all $n$ clients are active, the DNS server sees at most one query per $T$ seconds from that LDNS.

## Load Granularity

The **load granularity** $G = \bar{s}/\lambda$ is the expected traffic volume committed per DNS decision, where $\bar{s}$ is the mean transfer size:

```math
\begin{align}
  G &= \bar{s} \quad \text{(non-caching)} \\
  G &= n r \bar{s} T \quad \text{(caching with TTL } T \text{)}
\end{align}
```

A large $G$ means each DNS assignment commits a large traffic chunk, coarsening the effective granularity of load balancing. High $n$ (many clients per LDNS), large $\bar{s}$ (large files), or large $T$ (long TTL) all increase $G$ and worsen balance accuracy.

## Load Balancing Error Metric

Instantaneous load balance error between two links at time $t$ is defined as the normalized absolute difference in utilization:

```math
\begin{align}
  \varepsilon(t) = \frac{|U_1(t) - U_2(t)|}{U_1(t) + U_2(t)}
\end{align}
```

where $U_l(t) \in [0, 1]$ is the fraction of link $l$'s capacity currently utilized. The time-averaged error $\bar{\varepsilon}$ over an interval $I$ is the primary performance metric. Perfect balance corresponds to $\varepsilon = 0$; complete concentration on one link gives $\varepsilon = 1$.

> [!NOTE]
> The paper uses an averaging interval $I \geq 15$ seconds for computing $\bar{\varepsilon}$, because shorter intervals show high variance from bursty TCP connections. At $I = 15$ s the metric stabilizes and is representative of sustained imbalance.

# Factors Affecting DNS-ITE Accuracy

## Aggregate Load

Higher aggregate utilization reduces error because links approach capacity and the MB algorithm's decisions matter more. As aggregate utilization increases from low to medium, error decreases roughly 20–30% — the system has fewer opportunities to make disproportionate assignments when both links are already loaded.

## Transfer Size Distribution

Larger mean transfer sizes increase error because a single DNS assignment commits more load. Fixed-size 625 KB transfers show 15–25% higher error than 30 KB transfers. Under a lognormal transfer size distribution (heavy-tailed), errors increase by another 10–20% compared to fixed-size equivalents, because occasional very large transfers cause unpredictable load spikes.

## Clients per LDNS

More clients behind a single LDNS server increases load granularity $G$ and increases error, because the system cannot sub-divide the LDNS's traffic across links. Doubling the clients per LDNS roughly doubles the per-assignment traffic burst, proportionally worsening balance.

## TTL and Caching

Short TTLs increase $\lambda$ and reduce $G$, allowing finer-grained assignment. When the per-client request rate $r$ satisfies $r > 1/T$ (clients request faster than the TTL), caching is the binding constraint and shorter TTLs improve balance. TTL violations by non-compliant LDNS servers effectively shorten $T$, inadvertently helping load balance accuracy.

## Measurement Window $W$

The optimal window balances staleness against variance:

| Window $W$ | Staleness | Variance | Behavior |
|---|---|---|---|
| $W = 0.1$ s | Very low | Very high | Oscillates between links; assigns based on noise |
| $W = 5$–$15$ s | Low | Low | Best accuracy; tracks sustained utilization |
| $W = 30$ s | High | Very low | Misses recent load changes; assigns to already-loaded links |

> [!IMPORTANT]
> The decision-to-traffic delay $\delta$ sets a hard lower bound on useful window size: measurements older than $\delta$ are already reflected in current utilization. Using $W \ll \delta$ wastes accuracy by measuring noise. In practice $\delta$ is on the order of seconds for TCP, consistent with the optimal $W \in [5, 15]$ s finding.

# Comparison with Related Approaches

| Method | Granularity | Measurement Needed | BGP Changes | TTL Sensitivity |
|---|---|---|---|---|
| BGP prefix de-aggregation | Per prefix (coarse) | No | Yes | N/A |
| BGP AS-path prepending | Per prefix (coarse) | No | Yes | N/A |
| DNS Round-Robin | Per DNS query | No | No | Low |
| DNS Measurement-Based (this paper) | Per DNS query | Yes | No | High |

DNS-based ITE avoids BGP complexity and operates at finer granularity than prefix-level BGP manipulation, but requires continuous traffic measurement and is sensitive to TTL violations and transfer size distribution. BGP-based approaches are more stable but cannot react quickly to short-term load fluctuations.

> [!TIP]
> Subsequent work on ITE includes SDN-based approaches (e.g., using OpenFlow to redirect specific flows) and anycast-based load balancing, which avoid the TTL caching problem entirely by selecting the ingress at the routing level rather than via DNS indirection.

# Experiments

- **Dataset**: 24-hour GTLIB (Georgia Tech software mirror) packet trace, April 10, 2008; 46,400 DNS A record resolutions; 2,864 unique LDNS servers identified; 92% client-LDNS association success rate
- **Hardware**: Apache web server on Linux; test clients from 40 PlanetLab nodes and 6 RON (Resilient Overlay Network) nodes
- **Averaging interval**: $I \geq 15$ seconds for stable $\bar{\varepsilon}$ estimates
- **Measurement window sweep**: $W \in \{0.1, 1, 5, 10, 15, 30\}$ seconds
- **Results**:
  - MB with $W \in [5, 15]$ s consistently outperforms Round-Robin
  - Error is insensitive to $W$ within the $[5, 15]$ s range, giving practical flexibility
  - Error decreases with aggregate load (higher utilization → better balance)
  - Lognormal file size distribution incurs 10–20% higher error than fixed-size transfers at the same mean
  - TTL compliance rate of 60% limits DNS query rate for compliant LDNS servers, increasing effective load granularity for those servers

# Applicability

**Who**: Network operators of multihomed data centers or content delivery infrastructure with multiple upstream ISP connections.

**When**: When traffic volumes are high enough that manual BGP tuning is insufficient, and fine-grained per-request load balancing is needed without deploying SDN or BGP route-reflector automation.

**Where**: Suitable for services where DNS query rates are at least on the order of one per $W$ seconds per active LDNS, so that measurement feedback remains timely. Less effective when all clients share a small number of LDNS servers with strict TTL compliance, since the effective re-assignment frequency is then bounded by $1/T$.

> [!CAUTION]
> The paper's conclusions are drawn from a single 24-hour trace at a single site (GTLIB). Generalization to other services (e.g., streaming video or real-time APIs with very different transfer size distributions or client counts per LDNS) should be done cautiously. The optimal window $W$ may differ for workloads with different connection duration distributions.
