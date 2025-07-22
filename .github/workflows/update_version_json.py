import sys
from pathlib import Path
import json

def update_version_json(version, site_url="https://dtsapetis.github.io/uqlm", json_path="versions.json"):
    tag = f"v{version}"
    latest_entry = {
        "name": f"{tag} (latest)",
        "version": version,
        "url": f"{site_url}/latest/"
    }

    path = Path(json_path)
    if path.exists():
        with open(path, "r") as f:
            versions = json.load(f)
    else:
        versions = []

    # Update existing versions to remove any 'latest' marker
    # and point URL to their specific tag folder
    for entry in versions:
        if "(latest)" in entry["name"]:
            entry["name"] = entry["name"].replace(" (latest)", "")
        # Always update URL to version-specific path (for all versions except the new latest)
        entry["url"] = f"{site_url}/v{entry['version']}/"

    # Insert new latest version at the beginning
    versions.insert(0, latest_entry)

    # Save changes
    with open(json_path, "w") as f:
        json.dump(versions, f, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python update_version_json.py <version> <json_path>")
        sys.exit(1)

    version = sys.argv[1]
    json_path = sys.argv[2]
    update_version_json(version, json_path=json_path)
