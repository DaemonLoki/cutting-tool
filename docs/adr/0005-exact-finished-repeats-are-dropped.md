# An exact repeat of a finished sentence is dropped

An earlier finished sentence is dropped when the next sentence has the same normalized words. The later sentence stays. Phase 1 treats that repetition as unintentional, including when the sentence is longer than `auto_drop_max_s` and when it crosses a source boundary. One such repetition produces one decision.

A finished sentence followed by different words still stays, flagged `complete_sentence`, and the judge cannot drop it. That covers a partial restart and a period the transcript adds to an aborted take. ADR 0004 still governs those cases.
