"""Field definitions and feature groupings for the session schema."""
from __future__ import annotations

# YAML frontmatter columns used as ML features
TEXT_FEATURES = ["title", "description"]

CATEGORICAL_FEATURES = [
    "track",
    "day",
    "session_type",
    "level_name",
    "conference",
    "status",
]

NUMERIC_FEATURES = [
    "level",
    "duration",
    "interest",
]

MULTI_LABEL_FEATURES = [
    "audience",
    "speakers",
    "tags",
]

ALL_FEATURES = TEXT_FEATURES + CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Valid target columns for supervised learning
VALID_TARGETS = ["track", "level", "conference", "level_name", "session_type", "day"]

# Day ordering for scheduling contexts
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

# Level code → name mapping
LEVEL_NAMES = {
    100: "Business Oriented",
    200: "Feature Oriented",
    300: "Technical",
    400: "Deep Technical",
}

# Known conference values
CONFERENCES = ["FABCON", "SQLCON"]

# Known tracks
FABCON_TRACKS = [
    "Admin and Governance",
    "Power BI",
    "Data Engineering",
    "Data Warehousing",
    "Data Science",
    "Data Integration",
    "Real-Time Intelligence",
    "OneLake",
    "Microsoft Purview",
    "Data Dev",
    "Developer Experiences",
    "Microsoft Foundry",
    "Power Platform",
]

SQLCON_TRACKS = [
    "SQL Server",
    "Azure SQL",
    "SQL in Fabric",
    "Cosmos DB",
    "PostgreSQL",
    "MySQL",
]

ALL_TRACKS = FABCON_TRACKS + SQLCON_TRACKS
