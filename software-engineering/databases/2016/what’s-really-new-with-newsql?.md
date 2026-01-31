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

ai

a

a