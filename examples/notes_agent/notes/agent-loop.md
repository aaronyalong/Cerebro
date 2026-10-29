# The agent loop

1. Send the conversation and tool schemas to the model.
2. If the reply contains tool calls, run them and send the results back.
3. Repeat until the model answers without calling tools, or a limit is hit.

Errors from tools should go back to the model as results, so it can fix its
own mistakes instead of crashing the run.
