# With a script, the closest take stays

With a script, the take closest to the script stays, even when it is earlier than an alternate. An alternate before that take is dropped. An alternate after it is dropped when `script.drop_later_takes` is true, and kept with a CHECK marker otherwise. The score gap between the chosen take and the alternate is stored on the decision. A gap below `script.min_score_gap` flags the drop `script_close_call`. Phase 1 guards do not apply to these decisions; the script is the evidence. This supersedes ADR 0004 for the scripted case only.

Without a script, Phase 1 rules are unchanged: the later take wins, an exact repeat of a finished sentence is dropped, and a finished sentence followed by different words stays.
