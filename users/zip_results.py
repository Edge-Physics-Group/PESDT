from pathlib import Path
import shutil
import zipfile
import sys


if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} '<run_pattern>'")
    print(f"Example: {sys.argv[0]} 'runD_1.4MW_*_YH_CCC'")
    sys.exit(1)


pattern = sys.argv[1]

# Find matching run directories
cases = sorted(
    p for p in Path(".").glob(pattern)
    if p.is_dir()
)

if not cases:
    print(f"No directories found matching '{pattern}'")
    sys.exit(1)

print("Found cases:")
for case in cases:
    print(f"  {case.name}")


# Name of the ZIP file
prefix = pattern.split("*")[0].rstrip("_")
zip_name = f"{prefix}_cases.zip"


json_files = []

try:
    # Create temporary renamed JSON files
    for case in cases:
        source = case / "output.json"

        if not source.exists():
            print(f"WARNING: {source} does not exist, skipping")
            continue

        destination = Path(f"{case.name}.json")

        shutil.copy2(source, destination)
        json_files.append(destination)

        print(f"Copied {source} -> {destination}")

    # Create ZIP containing ONLY the renamed JSON files
    with zipfile.ZipFile(
        zip_name,
        "w",
        compression=zipfile.ZIP_DEFLATED
    ) as zf:

        for json_file in json_files:
            zf.write(json_file, arcname=json_file.name)

finally:
    # Remove temporary JSON files
    for json_file in json_files:
        json_file.unlink(missing_ok=True)


print(f"\nCreated: {zip_name}")