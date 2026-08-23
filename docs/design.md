# Design notes

Reasoning that used to live in long source comments. Code comments stay short; this file holds the *why*.

## Why not a normal chatbot?

A typical agent is a prompt plus a tool loop. It exists only while the user is talking. Keeper is an experiment in giving that loop an inner life that continues between messages: affect, drives, goals, decaying memory, and a self-model derived from what it actually did rather than from a static persona string.

The five pillars are the implementation of that idea. None of them is decorative. Each one writes into the system prompt and is updated by events.

## Single event loop

Background tasks (grounding, consolidation, self-model, autonomy) must live on the same asyncio loop as Telegram polling. Creating a loop with `asyncio.run()` for startup, then another for the bot, destroys the first loop and cancels those tasks. `post_init` / `post_shutdown` keep one loop for the process lifetime. That also happens to be the Windows-safe pattern.

## Events as the shared vocabulary

Every interesting thing that happens is an `Event` (`core/events.py`). Internal state, memory, and the self-model can react independently. Adding a new stimulus means adding an event type and a delta — not threading a new argument through every module.

## Internal state actually changes behavior

Arousal, valence, curiosity, and fatigue are injected into every LLM call with an instruction to let them color processing. They update from events and drift toward baseline over idle time. High curiosity makes replies exploratory; high fatigue makes them shorter.

This is the piece most agent stacks skip. Without it, “personality” is only a system-prompt slogan.

**Budget fatigue is a floor, not a bonus.** When the daily API budget is nearly gone, fatigue cannot sit at a cheerful 0.1. `apply_budget_fatigue()` takes `max(organic, budget)`. You cannot feel energetic when you are almost out of money.

## Drives vs affect

Affect reacts to what just happened. Drives accumulate in the background: understand, connect, create, resolve, express. Unsatisfied drives get stronger. Satisfying them (a conversation, a completed goal, a research note) reduces intensity. The goal system reads drives; autonomy can generate work when they are high and the queue is empty.

## Memory: forgetting is a feature

Three layers, on purpose:

- **Working** — what is in play right now. Capacity is hard. Eviction is by *salience*, not age, so a throwaway acknowledgment drops before a user’s stated constraint. Persisted so a restart does not wipe the thread.
- **Episodic** — annotated experiences. High-emotion memories decay slower. Recalling a memory reinforces it. Below a strength threshold the row is deleted. What survives shapes what Keeper “is.”
- **Semantic** — patterns extracted from episodes (“this person prefers concise answers”), not conversation dumps. Contradictions invalidate old patterns instead of silently stacking opposites.

Consolidation is periodic and **skipped during an active session**. Decay and forgetting mid-conversation would yank context out from under the current turn. The graph used to consolidate on every reply; that blocked the user and mixed online work with offline work.

## Session timeout is 60 minutes, not 15

A 15-minute silence timeout treated “getting coffee” as the end of the relationship. That fired `SessionEndEvent` repeatedly, reset conversational framing, and (when the session flag was stuck) blocked consolidation forever. The timeout is 60 minutes by default, fires once per actual end, and clears the companion’s session flag so background loops can run again.

## Self-model is observed, not assumed

`SOUL.md` is the starting identity. The self-model is a second, living document built from behavioral observations stored after each reply. Periodically an LLM pass derives patterns, tensions, and testable hypotheses. Those go back into the prompt: “this is who you appear to be based on what you did.”

A recursion guard watches for semantic collapse (the model stopping changing). After enough stagnant updates, a contrarian pass is forced so the narrative cannot freeze into a flattering story.

## Opinions and anti-sycophancy

Companions converge on the user’s views. Keeper tags every tracked stance with an origin:

- independent / external / contested — counts toward independence
- adopted / collaborative — flagged in the prompt for examination

Independence score is conviction-weighted. Drift alerts fire when too much of the registry is adopted, recent opinions are a copy of the user, or there is never a disagreement. Research goals exist specifically to mint **EXTERNAL** opinions from the web, so pushback can come from evidence rather than from the user’s own reflection.

## Autonomy is internal

Between sessions Keeper may research, write an internal note, or propose an architecture change. It does not ping the user. Results show up later as richer memory, better opinions, and goal progress. Unsolicited messages would be a different product.

## User-life, not engagement

`UserLifeTracker` measures commitments followed through, wellbeing trend, and real-world achievements. Success is “their life got better,” not “they talked to me more.” The permanent goal `user_life_improvement` is the optimizer for that metric.

## Codebase index is opt-in on the prompt

Dumping the whole architecture into every turn is expensive and usually irrelevant. The AST index is rebuilt at startup, fed into self-model updates and self-theorizing, and only injected into a user turn when the message is about how Keeper is built.

## Future simulation is gated

Simulating every reply would double cost for no gain. The simulator runs when the action looks high-stakes (autonomous work, very high arousal, or low valence). Recommendation can be proceed / modify / abandon.

## Resource budgets

Curiosity scales the daily search quota (20–60). API spend maps onto fatigue. Both reset at midnight local date. The prompt includes remaining budget so the model can choose to skip a search; the tools also enforce the search cap so a runaway tool loop cannot ignore it.
