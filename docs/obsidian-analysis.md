# Obsidian Analysis Guide

How to use the Obsidian Bases and Canvases alongside the ML pipeline output.

## Bases Reference

### Built-in Bases

| Base | Purpose |
|------|---------|
| `Bases/All Sessions.base` | All sessions, grouped by Status/Interest/Track/Day |
| `Bases/My Attending Sessions.base` | Your selected sessions by day |
| `Bases/Considering Sessions.base` | Sessions in "Considering" status |
| `Bases/High Interest Sessions.base` | Sessions with interest ≥ 4 |
| `Bases/Session Conflicts.base` | Detect overlapping time slots |
| `Bases/Track Sessions.base` | Sessions for the embedding track page |
| `Bases/Speaker Sessions.base` | Sessions for the embedding speaker page |
| `Bases/All Speakers.base` | Speaker list |
| `Bases/All Tracks.base` | Track list |

### ML Analysis Bases (added by this project)

| Base | Purpose |
|------|---------|
| `Bases/Session Clusters.base` | Sessions grouped by `ml_cluster` (requires `--write-back`) |
| `Bases/Level Analysis.base` | Sessions by level with interest/status breakdown |
| `Bases/Sessions by Conference.base` | FABCON vs SQLCON comparison |
| `Bases/Speaker Network.base` | Speaker metadata with session counts |

## Using ML Results in Obsidian

### Step 1: Write cluster labels back to vault

```bash
eda cluster --vault . --algorithm kmeans --write-back
```

This adds `ml_cluster: 3` (for example) to each session's YAML frontmatter.

### Step 2: Create a Base view for clusters

The `Session Clusters.base` file groups sessions by the `ml_cluster` property, so you can
immediately see which sessions ended up in the same cluster.

Embed it in any markdown file:

```markdown
![[Session Clusters.base]]
```

### Step 3: Rate sessions in context

Open a cluster group — sessions in the same cluster are likely topically related.
Use the `interest` field to rate them 1–5, then filter on `Bases/High Interest Sessions.base`.

## Workflow Tips

- **Before the conference:** Sync data, set `interest` ratings, mark top picks as `Attending`.
- **Check conflicts:** `![[Session Conflicts.base]]` shows sessions in the same time slot.
- **Use Track pages:** Each `Tracks/` file embeds `Track Sessions.base` filtered to that track.

## Canvas Overview

### Session Analysis.canvas

Shows the analysis board connecting:
- Sessions data source → ML pipeline steps → scatter plot results
- Useful for presenting the analysis approach

### Speaker Network.canvas

Visualises speaker co-occurrence: nodes are speakers, edges connect speakers who appear
in the same session.

### ML Analysis.canvas

Flow diagram of the ML pipeline with result summary cards for each command.

## Embedding a Base in a Markdown File

```markdown
<!-- Default view -->
![[My Attending Sessions.base]]

<!-- Specific named view -->
![[My Attending Sessions.base#Wednesday]]

<!-- In a track overview page -->
![[Track Sessions.base]]
```

> **Requirement:** Enable the **Bases** core plugin in Obsidian Settings → Core plugins.
