import json
import os
import sys
from pathlib import Path

def rebuild_version_json(current_version, gh_pages_path, site_url="https://dtsapetis.github.io/uqlm"):
    version_json_path = Path(gh_pages_path) / "versions.json"
    entries = []

    # List only v* folders, ignore 'latest'
    folders = sorted([
        p for p in Path(gh_pages_path).iterdir()
        if p.is_dir() and p.name.startswith("v")
    ], reverse=True)  # newest first

    current_version_exists = False
    for folder in folders:
        version = folder.name[1:]  # strip leading 'v'
        entry = {
            "name": f"v{version}",
            "version": version,
            "url": f"{site_url}/v{version}/"
        }

        entries.append(entry)

    entries.append({
            "name": f"v{current_version} (latest)",
            "version": current_version,
            "url": f"{site_url}/latest/"
        })

    # Save version.json
    with open(version_json_path, "w") as f:
        json.dump(entries, f, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python rebuild_version_json.py <current_version> <gh_pages_path>")
        sys.exit(1)

    current_version = sys.argv[1]
    gh_pages_path = sys.argv[2]
    # os.chdir("..")
    # os.chdir("..")
    x = os.getcwd()
    rebuild_version_json(current_version, gh_pages_path)
