# LLM Council: Complete Documentation

## Overview

The LLM Council is a decision-making framework that routes questions through five independent AI advisors with different thinking styles, who then peer-review each other anonymously before a chairman synthesizes a final verdict.

## The Five Advisor Perspectives

**The Contrarian** actively hunts for fatal flaws and weaknesses, assuming the idea will fail and digging to prove it wrong.

**The First Principles Thinker** strips away surface-level framing to ask what problem is actually being solved, sometimes revealing the wrong question is being asked.

**The Expansionist** identifies upside potential and adjacent opportunities, focusing on what could exceed expectations rather than risk mitigation.

**The Outsider** brings zero domain expertise, catching blind spots and clarity issues that insiders miss due to curse of knowledge.

**The Executor** evaluates feasibility and immediate next steps, asking "what happens Monday morning?" and dismissing ideas without clear implementation paths.

## When to Trigger the Council

**Mandatory triggers:** "council this," "run the council," "war room this," "pressure-test this," "stress-test this," "debate this"

**Strong triggers:** Genuine decisions with tradeoffs like "should I X or Y," questions about validation, or expressions of uncertainty between meaningful options

**Don't trigger on:** Factual lookups, simple yes/no questions, or casual queries without real stakes

## The Process

1. **Frame the question** with enriched workspace context (scanning for relevant files like CLAUDE.md or memory folders)
2. **Convene advisors** spawning all five in parallel, each responding 150-300 words from their distinct perspective
3. **Peer review** anonymizing responses A-E, with each advisor reviewing the others and identifying strongest/weakest arguments and collective blind spots
4. **Chairman synthesis** producing a final verdict covering: areas of agreement, genuine disagreements, emerging blind spots, and a single concrete next step
5. **Generate outputs** as both an HTML report (for scanning) and markdown transcript (for reference)

## Key Principles

All advisors spawn simultaneously to prevent sequential bias. Anonymization during peer review ensures merit-based evaluation. The chairman may disagree with the majority if reasoning supports it. Trivial questions bypass the council entirely.
