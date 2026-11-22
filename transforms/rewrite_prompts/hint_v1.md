## Instructions
The agent above did not pass the task's tests. It will try again from one of its steps, on its own: the steps before
that step run again exactly as they were, and just before it the agent reads one short hint from you.

1. Find the first step where the agent went wrong: the earliest step from which a different action could still have
   led to success (a wrong belief, a wrong or missing command, a conclusion the terminal output did not support, giving
   up too early). Not step 1: from there the agent would simply start over.
2. Write the hint for that step: one sentence of at most {max_words} words that tells the agent what to check or
   reconsider. Give away as little as possible: no command, no code, no file contents, no values, no expected output,
   and do not say what the right approach or the answer is. Never mention the tests or their output. The agent has to
   find the fix itself.
   The level of detail to aim for: "Check what the input file actually contains before deciding how to parse it."
   or "Re-read which output format the task asks for."
3. If no step could have been done better (for example, the task cannot be solved in this environment), answer with
   "step": null.

## Answer: only a JSON object
{"step": <step number>, "hint": "<one sentence>"}
or
{"step": null, "hint": null}
