## What to change
1. Find where the agent went wrong: the command whose result made the tests fail, and the reasoning that led to it.
2. Correct only that: the wrong reasoning and the wrong commands, replaced by ones that produce exactly what the tests
   check. Prefer the agent's own approach, fixed, over a different approach of your own.
3. Write the terminal output every changed command would print, and update every later step that refers to the old
   output: its reasoning, the analysis and plan of its action, and its terminal output.
4. Delete turns that only followed the mistake and are no longer needed. You cannot add turns: put the commands a fix
   needs into the existing turns.

## What to keep
5. Everything else exactly as the agent wrote it: its correct steps, exploration, checks, hesitation ("maybe", "I
   think") and self-corrections, even when they look redundant. Do not clean up, shorten or polish, and do not change
   style, formatting or command preference. The result must stay the agent's own trajectory with its mistake corrected.
6. Steps {protected}: they complete the task. Keep their commands; you may correct their reasoning, the analysis and
   plan of their action, and their terminal output where an earlier change alters what it shows.

## No hindsight
7. You see the task's tests; the agent did not. Nothing the agent reasons or does may rely on them: every corrected
   step must follow from the task statement and from what the terminal had shown before that step. Do not write the
   expected output into a command or a statement unless the agent could have derived it at that point, and do not
   have the agent predict a command's exact output before the terminal shows it.
8. A terminal output you write must be what the real shell would print, in the same format as the surrounding
   outputs, including the first line ("New Terminal Output:") and the shell prompt lines.

## Decide
9. As small as possible: make the fewest and shortest edits that make the tests pass. All edits together may change at
   most {change_share} of the agent's text (its reasoning and actions; a deleted turn counts whole; the terminal
   outputs you write do not count).
10. Decide whether the rewritten trajectory passes the tests. If you cannot make it pass within the existing turns and
    that limit, answer with no edits and "passes": false.

## How to write the edits
- edit_thinking: "old" is a short exact quote from that step's reasoning, copied character for character (one sentence
  or less), that occurs exactly once there; "new" replaces it, and an empty "new" deletes it.
- set_command: the complete action JSON the agent writes: {"analysis": "...", "plan": "...", "commands":
  [{"keystrokes": "...\n", "duration": 0.1}], "task_complete": false}.
- set_terminal: the full terminal output after that step.

