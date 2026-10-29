# Model Context Protocol (MCP)

MCP is an open protocol for connecting models to tools and data sources.
A server exposes tools, resources and prompts; a client (the agent harness)
discovers them and calls them on the model's behalf.

Why it matters: one tool server works with any MCP-capable harness, instead
of rewriting the same integration for every framework.
