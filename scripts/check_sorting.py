import glob
import os
import re

raw_files = glob.glob("output/06_keyframes/*.jpg")
std_sorted = sorted(raw_files)
num_sorted = sorted(raw_files, key=lambda p: int(re.search(r"_f(\d+)", p).group(1)) if re.search(r"_f(\d+)", p) else 0)

print(f"Total keyframes: {len(raw_files)}")
print("\nStandard sorted first 15:")
for p in std_sorted[:15]:
    print(" ", os.path.basename(p))

print("\nNumeric sorted first 15:")
for p in num_sorted[:15]:
    print(" ", os.path.basename(p))

is_identical = [os.path.basename(a) for a in std_sorted] == [os.path.basename(b) for b in num_sorted]
print(f"\nAre std_sorted and num_sorted identical? {is_identical}")

if not is_identical:
    diff_count = sum(1 for a, b in zip(std_sorted, num_sorted) if a != b)
    print(f"Discrepancy count: {diff_count} out of {len(raw_files)}")
    for i, (a, b) in enumerate(zip(std_sorted, num_sorted)):
        if a != b:
            print(f"  First mismatch at index {i}: std='{os.path.basename(a)}' vs num='{os.path.basename(b)}'")
            break
