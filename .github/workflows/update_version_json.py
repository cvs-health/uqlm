import json
import sys
from pathlib import Path


def rebuild_version_json(current_version, gh_pages_path, site_url="https://cvs-health.github.io/uqlm"):
    version_json_path = Path(gh_pages_path) / "versions.json"
    entries = []

    # List only v* folders, ignore 'latest'
    folders = [p for p in Path(gh_pages_path).iterdir() if p.is_dir() and p.name.startswith("v")]

    folders = sorted(folders, key=lambda f: tuple([int(x) for x in f.name[1:].split(".")]), reverse=True)

    entries.append({"name": f"v{current_version} (latest)", "version": current_version, "url": f"{site_url}/latest/"})
    for folder in folders:
        version = folder.name[1:]  # strip leading 'v'
        entry = {"name": f"v{version}", "version": version, "url": f"{site_url}/v{version}/"}

        entries.append(entry)

    # Save version.json
    with open(version_json_path, "w") as f:
        json.dump(entries, f, indent=4)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python rebuild_version_json.py <current_version> <gh_pages_path>")
        sys.exit(1)

    current_version = sys.argv[1]
    gh_pages_path = sys.argv[2]
    rebuild_version_json(current_version, gh_pages_path)
