import json
import urllib.request

packages = [
    "ipython",
    "fsspec",
    "numpy",
    "pandas",
    "pyarrow",
    "adbc-driver-manager",
    "hypothesis",
    "pytest",
    "pytest-xdist",
    "bottleneck",
    "numba",
    "numexpr",
    "scipy",
    "xarray",
    "s3fs",
    "gcsfs",
    "odfpy",
    "openpyxl",
    "python-calamine",
    "pyxlsb",
    "xlrd",
    "xlsxwriter",
    "pyiceberg",
    "tables",
    "pyreadstat",
    "sqlalchemy",
    "psycopg2",
    "adbc-driver-postgresql",
    "pymysql",
    "adbc-driver-sqlite",
    "beautifulsoup4",
    "html5lib",
    "lxml",
    "matplotlib",
    "jinja2",
    "tabulate",
    "pyqt5",
    "qtpy",
    "zstandard",
    "pytz",
    "fastparquet",
]

seen = set()
for pkg in packages:
    if pkg in seen:
        continue
    seen.add(pkg)

    pypi = f"https://pypi.org/project/{pkg}/"
    gh = ""
    err = ""

    try:
        with urllib.request.urlopen(f"https://pypi.org/pypi/{pkg}/json", timeout=20) as resp:
            data = json.load(resp)

        info = data.get("info", {})
        proj = info.get("project_urls") or {}
        home = info.get("home_page") or ""

        for key in ("Source", "Source Code", "Repository", "Code", "GitHub", "Homepage", "Home"):
            val = proj.get(key)
            if isinstance(val, str) and "github.com" in val.lower():
                gh = val
                break

        if not gh and isinstance(home, str) and "github.com" in home.lower():
            gh = home

        if not gh:
            for val in proj.values():
                if isinstance(val, str) and "github.com" in val.lower():
                    gh = val
                    break
    except Exception as ex:
        err = str(ex)

    print(f"{pkg}\t{pypi}\t{gh}\t{err}")
