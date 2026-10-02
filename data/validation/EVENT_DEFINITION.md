# What counts as one event (labelling rules)

These rules define the ground truth for the evaluation days in `eval_days/`. Every metric depends on them, so apply them the same way on every day. If a rule turns out to be wrong, change it here and re-check the days already labelled.

## The rule
Two articles belong to the **same event** when they report the **same real-world incident**: the same main actors, the same place (if any) and the same time window. Wording, outlet and angle don't matter.

The same event also includes:
- **follow-ups** to that incident within a few days (an arrest after a crime, a result after a vote, a reply to a statement)
- **reactions, analysis and explainers** that are *mainly* about that incident
- **live pages, previews and wrap-ups** of that incident

These are **different events**:
- the same person or organisation in a **different** incident (a cricketer's quote on Monday vs. his innings on Wednesday)
- the same **topic or storyline** but a different incident (two matches in one series, two court hearings in one case on different days with different outcomes, two separate floods)
- a scheduled thing vs. an unrelated thing that happens on the same day

## Cricket and sport
- **One match is one event:** preview, toss, live score, result, player reactions and match analysis.
- A **series or tournament is a storyline, not an event.** "Squads announced for the Asian Games" is its own event; each match is another.
- "Where to watch / live streaming" pages belong to the match they cover.
- Team selection or an injury announcement is its own event, unless it is reported only as part of one match's coverage.

## Politics, business, other
- A speech, statement or press conference is one event, together with direct responses to it within a few days.
- A policy or bill: announcement, passage and implementation on different days are **different events** in one storyline. Reporting on the *same* step is one event.
- Earnings, deals and market moves: one company's one announcement is one event. Daily market wraps are singletons unless two outlets wrap the same session.

## Singletons and unclear cases
- **Round-ups** covering several incidents ("Top 10 news today", "live updates" blogs spanning many stories) and **evergreen features** (profiles, old interviews, how-to) get their own event (a singleton).
- If you can't tell from the headline and description, pick your best judgement and **add a note**. Notes are counted later to see how often the rule is unclear.

## Storyline note (not used for scoring)
The labelling page has a free-text note field. You can write a storyline there (e.g. `storyline: Asian Games women's cricket`). It isn't scored now, but it records the hierarchy for a later "storyline" layer (spec §9).
