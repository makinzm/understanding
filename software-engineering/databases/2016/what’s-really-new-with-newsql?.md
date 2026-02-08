# Meta Information

- URL: [What's Really New with NewSQL? | ACM SIGMOD Record](https://dl.acm.org/doi/10.1145/3003665.3003674)
- LICENSE: [Publication Rights & Licensing Policy](https://www.acm.org/publications/policies/publication-rights-and-licensing-policy)
  - Published in 2016, before ACM's Open Access transition (January 1, 2026).
  - ACM Copyright Transfer Agreement applies.
    - [acm.org/binaries/content/assets/publications/copyreleaseproc-8-16.pdf](https://www.acm.org/binaries/content/assets/publications/copyreleaseproc-8-16.pdf)
  - Note: Freely accessible in ACM Digital Library Basic version since 2026, but original copyright terms remain.

```bibtex
@article{pavlo2016s,
  title={What's really new with NewSQL?},
  author={Pavlo, Andrew and Aslett, Matthew},
  journal={ACM Sigmod Record},
  volume={45},
  number={2},
  pages={45--55},
  year={2016},
  publisher={ACM New York, NY, USA}
}
```

# Abstract

Conventional DBMSs can do OLTP well but do not scale out.

NewSQL promotes scale OLTP workloads on distributed systems while maintaining ACID guarantees.

In this paper, we describe the history and architecture of NewSQL systems.

# History of DBMSs

- 1960s: IBM's IMS, DBMS for Apollo program.
- 1970s: 
  - IBM's System R and the University of California's INGRES.
  - Oracle DBMS.
- 1980s: Other commercial DBMSs: Sybase and Informix.
- 1990s: 
  - Object-oriented DBMSs
  - XML DBMSs
  - NoSQL systems.
  - MySQL and PostgreSQL.
- 2000s:
  - Struggle to scale DBMSs for web applications.
  - To overcome limitations of traditional DBMSs, custom middleware to shard single-node DBMSs.
    - eBay's Oracle-based cluster and Google's MySQL-based cluster.
  - Some companies built distributed DBMSs because of MySQL's performance limitations.
  - NoSQL systems emerged including key-value, graph or document stores.
    - Examples: Google's Bigtable and Amazon's Dynamo.
    - And then Facebook's Cassandra, PowerSet's HBase and MongoDB.
  - NoSQL cannot support strong consistency and ACID guarantees so not suitable for buisness applications.

# NewSQL Systems

NewSQL is "a class of modern relational DBMSs that seek to provide the same scalableperformance of NoSQL for OLTP read-write workloads whilestill maintaining ACID guarantees for transactions."

The difference between NewSQL and OLAP DBMSs is the purpose of the system, where OLAP are designed for large-scale analytical queries while NewSQL are designed for high-throughput transactional workloads.

# Categories of NewSQL Systems

1. New architectures built completely from scratch
2. Middleware rebuilt using transparent sharding
3. DBaaS offerings

## New Architectures

- Features:
  1. "Multi-node concurrency control"
  2. "Fault tolerance throught replication"
  3. "Flow control"
  4. "Distributed query processing"
- Examples: Clustrix, Google Spanner, H-Store, Hyper, MemSQL, NuoDB, SAP HANA, VoltDB

## Transparent Sharding Middleware

- Features:
  1. "Split a database into multiple shards"
  2. Centralized component management: Query routing, Transaction coordination, Replication management
- Examples: AgileData Scalable Cluster, MariaDB MaxScale, ScaleArc, ScaleBase

# The state of the art

## Main Memory Storage

- Past DBMSs use block-based storage like an SSD or HDD, so it is slow in I/O operations.
- Modern memory is affordable enough to store entire OLTP databases in RAM, eliminating need for buffer pools and heavy concurrency control.
- Larger-than-memory support: Systems can evict cold data to disk while keeping hot data in memory
- Challenge: Must retain keys of evicted tuples in indexes, limiting memory savings for apps with many secondary indexes.

## Partioning / Sharding

- This concept is not new, but was not addapted due to expensive hardwaren and low transaction demands.
- New Methods:
  - Horizontal Partitioning
  - Range / Hash Partitioning
- OLTP Optimization: Tree structures for patition so the operations can be done in one partition and reduce cross-partition transactions.
- Heterogeneous Architectures: Use different hardware for different partitions based on access patterns not homogenous hardware.

## Concurrency Control

- Architecture types:
  - Centralized coordinator: All operations go through single coordinator (like 1970s-80s TP monitors)
  - Decentralized: Each node maintains transaction state and coordinates with others; better scalability but requires highly synchronized clocks
- Historical schemes (1970s-80s): Two-phase locking (2PL) was dominant
- Modern NewSQL trend: Avoid 2PL due to deadlock complexity; use timestamp ordering (TO) variants instead
- Most common: Multi-Version Concurrency Control (MVCC):
  - Creates new tuple version on update; allows concurrent reads without blocking writes
  - Concept dates to 1979 MIT dissertation; first commercial use in 1980s
- Conclusion: Core algorithms aren't fundamentally new, but engineering optimizations for modern hardware and distributed environments are impressive

## Secondary Indexes

- Challenge in distributed DBMSs: Secondary indexes cannot always be partitioned the same way as primary data
  - Example: Tables partitioned by customer ID, but queries need to lookup by email address → would require broadcasting to all nodes (inefficient)
- Two design decisions:
  1. Where to store indexes
  2. How to maintain them in transactions
- Centralized coordinator approach (sharding middleware): Indexes can reside on coordinator and shard nodes; single version across system makes maintenance easier
- NewSQL decentralized approach: Use partitioned secondary indexes
  - Each node stores a portion of the index (not complete copy)
  - Trade-offs:
    - Partitioned indexes: Queries may span multiple nodes, but updates only modify one node
    - Replicated indexes: Lookups satisfied by single node, but updates require distributed transaction across all copies

> [!NOTE]
> In partitioned indexes, first lookup can find the id from e-mail via secondary index on one node, then select the data from id via primary index on another node.
>
> In replicated indexes, any node have the complete secondary index, so lookup and select can be done on the same node in the best case. However, update must be propagated to all nodes.

- Workaround for systems without native support: Developers use external distributed caches (e.g., Memcached), but application must manually maintain cache since DBMS won't auto-invalidate

## Replication

- Purpose: Ensure high availability and data durability for OLTP applications
- DBaaS advantage: Hide complexity of replication setup; customers don't worry about log transmission or node synchronization

### Two Design Decisions

#### 1. Data Consistency Model

Strong Consistency (all NewSQL systems use this):
- Transaction writes must be acknowledged and installed at all replicas before commit
- Pros:
  - Replicas can serve read-only queries with consistency guarantee
  - No lost updates on replica failure (all nodes synchronized)
- Cons:
  - Requires atomic commitment protocol (e.g., two-phase commit)
  - Additional overhead and potential stalls on node failure or network partition
- Contrast with NoSQL: NoSQL uses eventual consistency (weak consistency) - writes succeed without all replicas acknowledging

> [!NOTE]
> Cassandra selects consistency level per operation like ONE, QUORUM, ALL, LOCAL_QUORUM, EACH_QUORUM and so on.
> 
> However, NewSQL systems always use strong consistency for transactions because they need to guarantee ACID properties.

#### 2. Replication Execution Model

Active-Passive (most NewSQL systems):
- Request processed at single node first, then state transferred to replicas
- Required because non-deterministic concurrency control means queries might execute in different order on replicas
- Order depends on network delays, cache stalls, clock skew

Active-Active (deterministic DBMSs only):
- All replicas process same request simultaneously
- Used by: H-Store, VoltDB, ClearDB
- Possible because deterministic execution guarantees same operation order across replicas
- Systems must prevent queries using external/non-deterministic data (e.g., local system clock timestamps)

> [!NOTE]
> In active-active replication, it is dangerous to use non-deterministic functions like NOW() or RANDOM() because different nodes may get different results, leading to divergence.
>
> I think this pattern should not be used in life-dangerous systems like banking systems or flight control systems.

### Wide-Area Network (WAN) Replication

- New consideration: Modern deployments span multiple geographically distributed data centers
- Challenge: Synchronous updates over WAN cause significant slowdown
- Common approach: Asynchronous replication

## Crash Recovery

- Goal shift: Beyond preventing data loss (traditional focus), modern DBMSs must minimize downtime
  - Modern web applications expected to be online 24/7; outages are costly

### Traditional Single-Node Recovery

Process:

1. DBMS comes back online after crash
2. Loads last checkpoint from disk
3. Replays write-ahead log (WAL) to restore database state to moment of crash

### Distributed DBMS Recovery Challenge

Problem: Traditional approach doesn't work directly in distributed systems with replicas

Scenario:

1. Master node crashes
2. System promotes slave node to new master
3. New master continues processing transactions → database state moves forward
4. Previous master comes back online
5. ❌ Cannot just load checkpoint + replay WAL (would be out of sync)

Solution needed: Recovering node must get updates it missed from new master and other replicas

### Two Recovery Approaches

#### Approach 1: Incremental Catch-Up

1. Recovering node loads its last checkpoint + WAL from local storage
2. Pulls missing log entries from other nodes
3. Applies log updates faster than new updates arrive
4. Eventually converges to same state as other replicas

Requirements:
- Works if DBMS uses physical or physiological logging
- Log application time << original SQL execution time
- Node can "catch up" to current state

#### Approach 2: Full Snapshot

1. Recovering node discards its old checkpoint
2. System takes new snapshot from current state
3. Node recovers from fresh snapshot

Benefits:
- Faster recovery time
- Same mechanism can add new replica nodes to cluster

### Implementation in NewSQL Systems

Middleware & DBaaS: 
- Rely on underlying single-node DBMS recovery mechanisms
- Add infrastructure for leader election and management

New Architecture NewSQL:
- Combination of off-the-shelf components (ZooKeeper, Raft)
- Custom implementations of existing algorithms (Paxos)
- Standard procedures/technologies available since 1990s

## Future Trends: HTAP (Hybrid Transaction-Analytical Processing)

- Next trend: Execute analytical queries and ML algorithms on freshly obtained data
- "Real-time analytics": Analyze combination of historical + new data to extract insights
- Key difference from traditional BI: Can analyze new data immediately, not just historical data
- Why it matters: Data has immense value when created, but value diminishes over time

### Three Approaches to HTAP

#### Approach 1: Separate DBMSs (Most Common)

Architecture:

```bash
Front-end OLTP DBMS (stores new transaction data)
         ↓
   ETL (Extract-Transform-Load) utility
         ↓
Back-end Data Warehouse DBMS (executes OLAP queries)
         ↓
Results pushed back to front-end DBMS
```

#### Approach 2: Lambda Architecture

Architecture:

```bash
Batch Processing System (e.g., Hadoop, Spark)
  - Computes comprehensive view on historical data
  - Periodically rescans dataset
  - Bulk uploads results to stream processor

Stream Processing System (e.g., Storm, Spark Streaming)
  - Provides views of incoming data
  - Makes modifications based on new updates
```

Problems with Approaches 1 & 2 (Bifurcated Environment):

1. High latency: Data propagation measured in minutes or hours
   - Inhibits ability to act on data immediately
2. Administrative overhead: Deploying/maintaining two DBMSs is non-trivial
   - Personnel costs ~50% of total ownership cost for large-scale systems
3. Developer complexity: Must write queries for multiple systems to combine data
4. Hidden split systems: Some platforms (e.g., Splice Machine) hide architecture but still copy data from OLTP (HBase) to OLAP (Spark), causing technical issues

#### Approach 3: Single HTAP DBMS (Better Approach)

Characteristics:
- Single system supporting both:
  - High throughput + low latency for OLTP
  - Complex, long-running OLAP queries on hot (transactional) + cold (historical) data
- Innovation: Incorporates last decade's advancements from specialized systems:
  - OLTP: In-memory storage, lock-free execution
  - OLAP: Columnar storage, vectorized execution
  - All within single DBMS

### HTAP Implementation Examples

SAP HANA (first to market as HTAP):
- Multiple execution engines internally
- Row-oriented engine for transactions
- Column-oriented engine for analytical queries

MemSQL:
- Two storage managers (one for rows, one for columns)
- Single execution engine mixes both

HyPer:
- Switched from row-oriented + H-Store-style concurrency control (OLTP-focused)
- To column-store architecture + MVCC (supports complex OLAP)

VoltDB:
- Pivoted marketing from pure OLTP performance to streaming semantics

S-Store:
- Adding stream processing operations on top of H-Store architecture

Future prediction: Specialized OLAP systems from mid-2000s (e.g., Greenplum) will add better OLTP support

### Future Outlook

Not the end of giant OLAP warehouses:
- Still necessary short-term as universal back-end for organization's front-end OLTP silos

Long-term vision - Database Federation Resurgence:
- Execute analytical queries spanning multiple OLTP databases (even multiple vendors)
- Without needing to move data around
