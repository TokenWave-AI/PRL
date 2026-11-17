## What to change
1. Find where the agent went wrong: the command whose result made the tests fail, and the reasoning that led to it.
2. Correct that reasoning and replace the wrong commands with ones that produce exactly what the tests check. Prefer
   the agent's own approach, fixed, over a different approach of your own.
3. Write the terminal output every changed command would print, and update every later step that refers to the old
   output: its reasoning, the analysis and plan of its action, and its terminal output.
4. Delete turns that only followed the mistake and are no longer needed. You cannot add turns: put the commands a fix
   needs into the existing turns.
5. Where it costs little, also clean up as for a successful trajectory: repeated work, dead ends, wrong statements.

## What to keep
6. Everything the agent did right, in its own words. No changes of style, formatting or command preference.
7. Steps {protected}: they complete the task. Keep their commands; you may correct their reasoning, the analysis and
   plan of their action, and their terminal output where an earlier change alters what it shows.

## No hindsight
8. You see the task's tests; the agent did not. Nothing the agent reasons or does may rely on them: every corrected
   step must follow from the task statement and from what the terminal had shown before that step. Do not write the
   expected output into a command or a statement unless the agent could have derived it at that point, and do not
   have the agent predict a command's exact output before the terminal shows it.
9. A terminal output you write must be what the real shell would print, in the same format as the surrounding
   outputs, including the first line ("New Terminal Output:") and the shell prompt lines.

## Decide
10. Decide whether the rewritten trajectory passes the tests. If you cannot make it pass within the existing turns,
    answer with no edits and "passes": false.

## How to write the edits
- edit_thinking: "old" is a short exact quote from that step's reasoning, copied character for character (one sentence
  or less), that occurs exactly once there; "new" replaces it, and an empty "new" deletes it.
- set_command: the complete action JSON the agent writes: {"analysis": "...", "plan": "...", "commands":
  [{"keystrokes": "...\n", "duration": 0.1}], "task_complete": false}.
- set_terminal: the full terminal output after that step.

