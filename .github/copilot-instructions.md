# Copilot Instructions for FabCon 2026 Vault

## Notes

- pip doesn't work in this repo, use `uv pip` instead
- powershell is used
- ensure venv is activated before running python commands (check for `(Fabcon2026)` in terminal prompt)

## Project Purpose

This is an Obsidian vault for analyzing sessions and speakers at **FabCon 2026** and **SQLCON 2026** (co-located conferences in Atlanta, March 16-20, 2026). It contains 267 sessions/workshops, 358 speaker profiles, and 19 tracks with dynamic views powered by **Obsidian Bases** (a core plugin).

## Architecture

### Key Components

- **Sessions** & **Workshops** - Individual markdown files with YAML frontmatter containing metadata (speakers, track, level, time, room, interest, status)
- **Bases** - Dynamic views (stored as `.base` YAML files) that filter/group/sort content by properties like status, interest, track, and day
- **Planning files** - Dashboard, personal schedule, and conflict detection
- **Speakers** - Speaker profile files with bios
- **Tracks** - Track overview pages that embed session views via Bases

### Data Flow

1. **Source of truth:** Sessionize APIs return authoritative session/speaker/room data
2. **Session files** are generated from API data, stored with standardized YAML frontmatter
3. **Bases** query the session files using filters on frontmatter properties
4. **Planning files** embed Bases to show personalized views

### Obsidian Bases Fundamentals

- `.base` files use **YAML** syntax, not SQL
- Bases are embedded in markdown using `![[FileName.base]]` or `![[FileName.base#ViewName]]` syntax
- Filters can use logical operators (`and`, `or`), property comparisons (`status == "Attending"`), and dynamic references (`file.hasLink(this.file)`)
- Views define display type (table), grouping (`groupBy: property`), and sort order
- Each view is independent; a single `.base` file can have multiple views

### Example Base Filter Pattern

```yaml
filters:
  and:
    - status == "Attending"
    - day == "Wednesday"
views:
  - type: table
    groupBy:
      property: start_time_24h
      direction: ASC
    order:
      - file.name
```

## Critical YAML Frontmatter Fields

### Sessions & Workshops

```yaml
title: "Session Title"
day: Wednesday                    # Used for scheduling views
start_time: "1:45 PM"             # 12-hour format (AM/PM)
start_time_24h: "13:45"           # 24-hour format (CRITICAL for sorting)
end_time: "2:45 PM"
duration: 60
room: "C111-C112"                 # Must not be empty; indicates scheduled status
track: "[[Power BI Track]]"       # Wiki link to track file
level: 200                        # 100=Business, 200=Feature, 300=Technical, 400=Deep
level_name: "Feature Oriented"
speakers:
  - "[[Speaker Name]]"            # Wiki links to speaker files
conference: FABCON                # FABCON or SQLCON
status: Considering               # Considering | Attending | Skip (user choice)
interest: 3                       # 1-5 rating (user preference)
session_type: "Breakout Session"  # Breakout, Workshop, CORENOTE, Sponsor Speaker, etc.
```

## Core Workflows

### For Session/Workshop Updates

1. Fetch data from Sessionize API endpoints:
   - Sessions: `https://sessionize.com/api/v2/1op0w2v7/view/All`
   - Workshops: `https://sessionize.com/api/v2/coqpz3x7/view/All`
2. Parse JSON response (sessions, speakers, rooms, categories arrays)
3. Build lookup maps for rooms, speakers, and category types
4. Generate markdown files with YAML frontmatter for each session
5. **Validate** with `Scripts/patch_start_times.py` to ensure `start_time_24h` field exists on all files
6. **Verify counts:** API returns 267 total (245 sessions + 22 workshops as of Jan 2026)

### For Adding/Modifying Planning Views

1. Create or edit `.base` file in `Bases/` directory
2. Use YAML syntax to define filters and views
3. For dynamic filters (embedding file context), use `file.hasLink(this.file)` pattern
4. Test embedding in a markdown file: `![[FileName.base#ViewName]]`
5. Verify the Base renders in Obsidian with correct data

### For Patching Session Files

Run `Scripts/patch_start_times.py` after importing new sessions to add `start_time_24h` field:

```powershell
cd /path/to/vault
uv run Scripts/patch_start_times.py
```

This script safely skips files that already have the field, making it safe to run multiple times.

## Validation & Verification

After any data sync or import:

- [ ] Session count matches API count (expect 267 total)
- [ ] All sessions have `room:` value (not empty)
- [ ] All sessions have `speakers:` with wiki links `[[Name]]`
- [ ] All sessions have `track:` with wiki link
- [ ] All sessions have `## Description` content
- [ ] `start_time_24h` field exists on all session/workshop files
- [ ] Bases render correctly in Obsidian (enable Bases core plugin)
- [ ] Planning views show expected data

## Tool Preferences

- **Data scraping:** Prefer Python with `requests` + `BeautifulSoup` over browser automation (faster, more reliable)
- **Session generation:** Use the pattern in `CLAUDE.md` under "Step 4: Create Python Scripts" as reference for creating speaker/session files from Sessionize API
- **File creation:** Use Python for batch markdown generation; the `.base` files are typically hand-edited for specific views

## Session File Validation Checklist

A properly populated session/workshop file includes:

- Non-empty `room:` value
- `speakers:` array with at least one `[[Speaker Name]]` wiki link
- `track:` with exactly one `[[Track Name]]` wiki link
- `## Description` section (not just frontmatter)
- Correct `day` value matching the conference schedule
- `start_time_24h` field in 24-hour format (HH:MM)

## Conferences in This Vault

| Conference | Tracks | Focus |
|---|---|---|
| **FABCON** | 13 tracks | Microsoft Fabric, Power BI, Data Engineering, Data Science, etc. |
| **SQLCON** | 6 tracks | SQL Server, Azure SQL, PostgreSQL, MySQL, Cosmos DB, SQL in Fabric |

## Conference Schedule

| Day | Date | Type |
|---|---|---|
| Monday | March 16 | Full-day Workshops |
| Tuesday | March 17 | Full-day Workshops |
| Wednesday | March 18 | Keynote + Breakout Sessions |
| Thursday | March 19 | Breakout Sessions + Expo |
| Friday | March 20 | Breakout Sessions |

## Dependencies & Scripts

### `src/eda/` Python Package

Located at `src/eda/`, this is a data analysis package for session/speaker processing:

- **Python requirement:** >=3.12
- **Build system:** Hatchling
- **Key dependencies:** pandas, scikit-learn, click, rich, python-frontmatter
- **Optional extras:** umap, hdf5, sql, excel, parquet, aws, gcp, xml, html, performance

To install in development:
```powershell
cd src/eda
uv pip install -e .
```

Run CLI tool:
```powershell
uv run eda --help
```

### `Scripts/patch_start_times.py`

Utility to add/update `start_time_24h` field on all session and workshop files. Safe to run multiple times.

```powershell
uv run Scripts/patch_start_times.py
```

## Key Planning Files

- `Planning/Conference Dashboard.md` - Overview, stats, quick links to tracks
- `Planning/My Schedule.md` - Day-by-day view of sessions marked as "Attending"
- `Planning/Session Conflicts.md` - Identify overlapping time slots in your selections

## Useful Base Files Reference

| Base File | Purpose | Context |
|---|---|---|
| `Track Sessions.base` | Shows sessions in a track (uses `this.file` context) | Embedded in Track overview pages |
| `Speaker Sessions.base` | Shows sessions by a speaker (uses `this.file` context) | Embedded in Speaker profile pages |
| `My Attending Sessions.base` | User's selected sessions by day | Referenced in `Planning/My Schedule.md` |
| `All Sessions.base` | All sessions grouped by Status/Interest/Track/Day | Dashboard view |
| `Session Conflicts.base` | Sessions by time slot for conflict detection | Conflict resolution tool |

## Notes for Maintainers

- **Before conference:** Sync data weekly as the schedule may change
- **During conference:** Update for room changes and cancellations
- **After conference:** Preserve session files; add recordings/resources as new fields
- **Unverified sessions:** Place incomplete/unscheduled sessions in `Sessions/_Unverified/` or `Workshops/_Unverified/`
- **Empty room fields:** Indicate the session hasn't been scheduled yet; re-sync closer to the conference
- **Wiki link precision:** Speaker and track names must match exactly; use exact string matching when generating links
