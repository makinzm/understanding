- [ 生成AIでスクラムによる開発はどう変わるか | Ryuzee.com ]( https://slide.meguro.ryuzee.com/slides/127 )
- LICENSE: Copyright Ryutaro Yoshiba (Ryuzee)

Reference

- [ DORA | State of AI-assisted Software Development 2025 ]( https://dora.dev/research/2025/dora-report/ )
    - [ 2025_abridged_state_of_ai_assisted_software_development_ja.pdf ]( https://services.google.com/fh/files/misc/2025_abridged_state_of_ai_assisted_software_development_ja.pdf )
    - [ 2025_state_of_ai_assisted_software_development.pdf ]( https://services.google.com/fh/files/misc/2025_state_of_ai_assisted_software_development.pdf )

---

# How Generative AI Is Transforming Scrum-Based Development

## Executive Summary

Generative AI has moved from a pilot-stage curiosity to a default tool inside software teams. For teams practicing Scrum, this shift does more than speed up coding — it changes which activities are bottlenecks, which artifacts deserve investment, and how the team should be shaped. This report applies the SCQA framework (Situation, Complication, Question, Answer) to summarize how Scrum-based development is being transformed, and what teams should do next.

## Situation — The Established World of Scrum

For the past two decades, Scrum has been the dominant framework for iterative software development. A Scrum team — comprising a Product Owner, Scrum Master, and Developers — works in time-boxed Sprints of up to one month, delivering a valuable, usable Increment each cycle through the three pillars of transparency, inspection, and adaptation.

Within this framework, a familiar time allocation emerged. Teams treated implementation as the scarcest resource and designed everything else around protecting it:

- **Implementation was the rate-limiting step.** Most Sprint capacity was consumed by writing code.
- **Parallelization was the lever.** More developers writing in parallel was assumed to yield more throughput.
- **Heavy up-front preparation was rational.** Estimation, refinement, and task decomposition existed to protect expensive implementation time from risk and rework.
- **Velocity and estimation mattered.** Teams measured velocity to avoid overloading Sprints and to forecast delivery.

In short, the entire optimization logic of traditional Scrum rested on one assumption: writing code is slow and expensive, so minimize waste around it.

## Complication — Generative AI Breaks the Core Assumption

Generative AI is now embedded in everyday development work. According to the *State of AI-assisted Software Development 2025* survey cited in the presentation, roughly 90% of respondents use AI as part of their job, and over 80% report productivity gains. Tools such as GitHub Copilot, Claude Code, and Cursor have compressed implementation time by an order of magnitude; ChatGPT, Gemini, and Claude have similarly compressed research time for APIs and libraries.

The consequence is that implementation is no longer the bottleneck. But new frictions have taken its place, and they do not live where traditional Scrum optimized:

- **Context-setting has become the new cost.** AI does not infer unstated intent. Ambiguous prompts or missing assumptions produce low-quality output and force repeated retries.
- **Verification has become the new bottleneck.** Roughly 30% of developers report trusting AI-generated code little or not at all, which places a heavy burden on critical review.
- **Parallel coding has lost its justification.** When a small team can generate large volumes of code quickly, the integration and coordination overhead of many parallel workers becomes pure drag.
- **Quality can collapse silently.** Large volumes of generated code against a poor codebase or vague inputs degrade quickly without strong safeguards.
- **Skill development is at risk.** If AI writes the code, junior developers lose the natural pathway through which senior reviewers are formed.

The traditional optimizations of Scrum — long refinement sessions, careful estimation, velocity tracking, parallel task assignment — were built for a world that no longer exists in the same form.

---

## Question — What Must Change in Scrum Practice?

If implementation is cheap and fast, but context, verification, alignment, and sustainability are now the constraints, the central question becomes:

> **How should Scrum teams restructure their time, artifacts, team composition, and ceremonies so that AI amplifies value delivery instead of amplifying risk?**

---

## Answer — Six Shifts for AI-Era Scrum

### 1. Reframe Documentation as AI Infrastructure

The Agile Manifesto preferred working software over comprehensive documentation — a reasonable stance when teams could rely on tacit shared understanding. AI has no such tacit memory. High-quality, text-based, version-controlled documentation is now the single most important input to AI-assisted development. "Garbage in, garbage out" is not a slogan; it is the governing constraint.

- Make prerequisites, inputs, and outputs explicit.
- Use examples and counter-examples (Specification by Example).
- Maintain consistent terminology with written definitions.
- Keep docs in text-based formats (Markdown) and in the repository itself (not only `AGENT.md` or `CLAUDE.md`) — architecture overviews, coding standards, ubiquitous language, Definition of Done, and more.
- Version-control everything, because AI runs are large-scale and often wrong; reversibility matters.

### 2. Shift from Parallel Coding to Mob Work in Smaller Teams

When code can be generated rapidly, the scarce resource is alignment, not typing speed. Mob programming — a small group working together on the same screen, reviewing AI output in real time — becomes the rational default. Reviews move from asynchronous inspection after the fact to synchronous collaboration as the work happens. Learning spreads across the team at the same rate the work proceeds.

Large teams lose their rationale: coordination costs grow, and mob work does not scale linearly past a handful of people. The presentation recommends splitting products by domain so that each team is small enough to be maintained by a few people, drawing on Team Topologies and dynamic re-teaming patterns (and, in some cases, more fluid structures like FAST).

### 3. Shorten Sprints and Refocus on the Sprint Goal

Faster implementation permits shorter Sprints — often one week or even a few days — which in turn increases feedback frequency. However, stakeholder availability for Sprint Reviews has a natural ceiling, so the internal development cycle and the Review cycle may decouple (for example, 2–3 day internal cycles with a bi-weekly Review).

A shorter Sprint demands a sharper Sprint Goal. Clear goals produce clearer documents, which produce better AI output, which accelerates completion. And because the goal — not the full backlog — is what must be achieved, teams can ship the minimum that satisfies the goal and discard features that the hypothesis proves wrong.

### 4. Redesign Backlog Items and Retire Heavy Estimation

Product Backlog Items no longer need to be sized to fit a single Sprint at the cost of clarity. Larger items are tolerable — provided the documentation, acceptance criteria, and test cases are rich enough for AI to implement correctly. Refinement becomes a mobbing activity focused on making items "AI-ready," often timed just before or even during Sprint Planning.

Because implementation is no longer the bottleneck, the value of fine-grained estimation — and of velocity as a planning tool — drops substantially. A lightweight roadmap is typically sufficient. The time saved on estimation should be reinvested in agreeing on what to build.

### 5. Adapt the Scrum Events

- **Sprint Planning:** Refinement can happen in the meeting itself via mob work. Task decomposition only needs to be detailed enough to reach shared understanding.
- **Daily Scrum:** When the team is mobbing continuously, the Daily Scrum is effectively happening all day. Its importance as a distinct event diminishes.
- **Sprint Review:** Stakeholder participation remains essential — AI does not change this. But stakeholders now bring AI-era expectations (faster prototypes, different risk concerns), and teams may need to realign expectations explicitly.
- **Sprint Retrospective:** "How we use AI" becomes a recurring, often standing, topic. Because AI tooling evolves rapidly, inspection and adaptation of AI practices must be frequent — improvement cannot wait for the Retrospective alone.

### 6. Keep Humans Accountable and Invest in Sustainability

Three problems from the field require deliberate investment:

- **Accountability cannot be delegated to AI.** "The AI said so" is never a justification. Decisions must be made by people, documented, and traceable. The Definition of Done should include explainability and record-keeping requirements, and AI-generated sections should be marked for inspection.
- **Quality varies with inputs and with the existing codebase.** A clean architecture, maintainable code, and unambiguous inputs are prerequisites for reliable AI output. Reviews, guidelines, templates, and automated tests matter more than before — not less.
- **Talent development cannot be sacrificed for short-term speed.** If AI writes the code, developers lose the day-to-day practice that once produced senior engineers. Organizations must deliberately carve out time for skill acquisition; managers who optimize only for near-term output erode the organization's long-term viability.

---

## Conclusion

Scrum's core principles — transparency, inspection, adaptation, and human accountability for the Sprint Goal — remain unchanged. What changes is the shape of the work around those principles. The old optimizations assumed scarce implementation capacity; the new optimizations must assume abundant implementation capacity and scarce alignment, context, and verification.

The practical playbook that emerges from this analysis is consistent: treat documentation as infrastructure, prefer mob work in small teams, shorten Sprints, center the Sprint Goal, redesign backlog items around AI-readiness and inspectability, retire heavy estimation, and invest continuously in learning. Teams that make these shifts will find that AI amplifies their delivery. Teams that do not will find that AI amplifies their existing problems at unprecedented speed.
