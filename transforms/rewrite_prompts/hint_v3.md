## Instructions
The agent above did not pass the task's tests. It will try again from one of its steps, on its own: the steps before
that step run again exactly as they were, and just before it the agent reads one short hint from you.

1. Find the first step where the agent went wrong: the earliest step from which a different action could still have
   led to success (a wrong belief, a wrong or missing command, a conclusion the terminal output did not support, giving
   up too early). Not step 1: from there the agent would simply start over.
2. Write the hint for that step: at most {max_words} words that point the agent at the part of the task, of its input or
   of its own output that it should look at again. Say where to look, not what is wrong there, and never how to fix
   it: no command, option, tool, code, file contents, value or expected output, and no "rather than" or "instead of"
   that implies the answer. Never mention the tests or their output. The agent has to find the mistake itself.
   Name only the part to look at (the task statement, the input, the agent's own output, the last command's
   output), never a detail inside it: no field, key, column, line, character, file name or format detail. The level of
   detail to aim for: "Re-check what the task says about the output." or "Look again at the input before parsing it."
3. If no step could have been done better (for example, the task cannot be solved in this environment), answer with
   "step": null.

## Answer: only a JSON object
{"step": <step number>, "hint": "<at most {max_words} words>"}
or
{"step": null, "hint": null}
