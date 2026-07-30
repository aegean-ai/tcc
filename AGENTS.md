# Agent Instructions

This project tracks issues in Jira, project AURA at https://aegean-ai.atlassian.net. Use the Atlassian MCP tools.

## Quick Reference

```text
Search issues   mcp__plugin_atlassian_atlassian__searchJiraIssuesUsingJql
View an issue   mcp__plugin_atlassian_atlassian__getJiraIssue
File an issue   mcp__plugin_atlassian_atlassian__createJiraIssue
Comment         mcp__plugin_atlassian_atlassian__addCommentToJiraIssue
Change status   mcp__plugin_atlassian_atlassian__transitionJiraIssue
```

## Non-Interactive Shell Commands

**ALWAYS use non-interactive flags** with file operations to avoid hanging on confirmation prompts.

Shell commands like `cp`, `mv`, and `rm` may be aliased to include `-i` (interactive) mode on some systems, causing the agent to hang indefinitely waiting for y/n input.

**Use these forms instead:**
```bash
# Force overwrite without prompting
cp -f source dest           # NOT: cp source dest
mv -f source dest           # NOT: mv source dest
rm -f file                  # NOT: rm file

# For recursive operations
rm -rf directory            # NOT: rm -r directory
cp -rf source dest          # NOT: cp -r source dest
```

**Other commands that may prompt:**
- `scp` - use `-o BatchMode=yes` for non-interactive
- `ssh` - use `-o BatchMode=yes` to fail instead of prompting
- `apt-get` - use `-y` flag
- `brew` - use `HOMEBREW_NO_AUTO_UPDATE=1` env var

<!-- BEGIN JIRA INTEGRATION -->
## Issue tracking with Jira

Issues live in Jira, project AURA at https://aegean-ai.atlassian.net. Use the Atlassian
MCP tools (`mcp__plugin_atlassian_atlassian__*`) for all issue work.

- Find work with `searchJiraIssuesUsingJql`, read with `getJiraIssue`, file with
  `createJiraIssue`, comment with `addCommentToJiraIssue`, move with `transitionJiraIssue`.
- Before filing, search for an existing issue covering the same work and update that instead.
- When you discover follow-up work mid-task, file it in Jira and link it to the issue you
  are working on rather than leaving a TODO in the code.
- Jira is the only tracker. Do not create markdown TODO lists or a second tracking system.
- There is no local issue database in this repo. beads and the `bd` CLI are retired: never
  run `bd`, never recreate a `.beads/` directory, and ignore any `.beads-archive.zip`.
<!-- END JIRA INTEGRATION -->

## Session completion

When you finish a chunk of work:

1. File follow-up work in Jira for anything left undone.
2. Run the quality gates if code changed (tests, linters, builds).
3. Update the status of the Jira issues you touched.
4. Leave the work committed or uncommitted according to what the user asked. Do not push on
   your own initiative: the user batches commits deliberately, and several repos protect
   `main` so direct pushes are rejected. Where a repo requires a PR, open one rather than
   pushing to `main`.
5. Hand off: say what is done, what is not, and what the next step is.
