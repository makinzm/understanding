# Software Engineering

Documentation of software engineering concepts, practices, and technologies across architecture, development processes, databases, performance, cloud infrastructure, and fundamentals.

## Directory Structure

Organized by category, with each category further organized by year to track evolving trends.

```
software-engineering/
├── architecture/       # Clean Architecture, design patterns, system design
│   └── <year>/
├── processes/          # Scrum, TDD, specification-oriented development, methodologies
│   └── <year>/
├── databases/          # NewSQL, SQL, NoSQL, data modeling
│   └── <year>/
├── performance/        # Optimization, profiling, benchmarking, near-hardware
│   └── <year>/
├── cloud/              # Cloud platforms, infrastructure, deployment, DevOps
│   └── <year>/
└── fundamentals/       # CS fundamentals, algorithms, data structures
    └── <year>/
```

### Document Template

Each document in this directory should follow this structure:

- **Meta Information**: Source URL, LICENSE, and Reference at the top
- **Sections**: Organized by the topic's natural structure
- **Concrete content**: Specific sentences demonstrating understanding (not "I understand X")
- **Applicability**: Who would use this, when, and where
- **Comparisons**: How this differs from similar concepts or alternatives

### File Naming

- Use kebab-case for all filenames (e.g., `clean-architecture.md`)
- Generate filenames with: `echo "Title" | bash scripts/title-converter.sh`
